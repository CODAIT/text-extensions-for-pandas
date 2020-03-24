#
#  Copyright (c) 2020 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# aggregate.py
#
# Gremlin traversal steps that perform aggregation
from abc import ABC
from typing import *

import pandas as pd

from text_extensions_for_pandas.gremlin.traversal.base import GraphTraversal,\
    UnaryTraversal
from text_extensions_for_pandas.gremlin.traversal.format import ValuesTraversal
from text_extensions_for_pandas.gremlin.traversal.underscore import __


class AggregateTraversal(UnaryTraversal, ABC):
    """
    Base class for all traversals that implement aggregation and return a single
    scalar value.
    """

    def pandas_agg_name(self) -> str:
        """
        :return: The name of the equivalent Pandas aggregation type
        """
        raise NotImplementedError("Subclasses must implement this method")


class SumTraversal(AggregateTraversal):
    """
    Gremlin `sum()` step.

    If applied to a span-valued input, combines the spans into a minimal span
    that covers all inputs.
    """
    def __init__(self, parent):
        UnaryTraversal.__init__(self, parent)

    def compute_impl(self) -> None:
        tail_type = self.parent.step_types[-1]
        if tail_type != "p":
            raise NotImplementedError(f"Sum of step type '{tail_type}' not "
                                      f"implemented")
        input_series = self.parent.last_step()
        self._set_attrs(
            paths=pd.DataFrame({0: [input_series.sum()]}),
            step_types=["p"],
            aliases={}
        )

    def pandas_agg_name(self) -> str:
        return "sum"


####################################################
# syntactic sugar to allow aggregates without __

def sum_():
    return __.sum()

# end syntactic sugar
####################################################


class GroupTraversal(UnaryTraversal):
    """
    Gremlin `group` step, usually modulated by one or more 'by` modulators.
    """

    def __init__(self, parent):
        UnaryTraversal.__init__(self, parent)
        self._key = None  # Type: str
        self._value = None  # Type: AggregateTraversal

    def by(self, key_or_value: Union[str, AggregateTraversal]) -> \
            "GroupTraversal":
        """
        `by` modulator to `group()`, for adding a key or value to the group
        step.

        Modifies this object in place.

        :param key_or_value: If a string, name of the field to group by; if
         an `AggregateTraversal`, subquery to run for each group.

        :returns: A pointer to this object (after modification) to enable
        operation chaining.
        """
        if isinstance(key_or_value, str):
            if self._key is not None:
                raise ValueError("Tried to set key of group() step twice")
            self._key = key_or_value
        elif isinstance(key_or_value, AggregateTraversal):
            if self._value is not None:
                raise ValueError("Tried to set value of group() step twice")
            self._value = key_or_value
        else:
            raise ValueError(f"Object '{str(key_or_value)}' passed to "
                             f"group().by() is of type '{type(key_or_value)}',"
                             f"which is not a supported type.")
        return self

    def compute_impl(self) -> None:
        tail_type = self.parent.step_types[-1]
        if tail_type != "v":
            raise NotImplementedError(f"Grouping on a path that ends in step "
                                      f"type '{tail_type}' not implemented")

        if (self._key is not None
                and self._value is not None
                # TODO: Parent must be values
                and isinstance(self._value.parent, ValuesTraversal)
                and self._value.parent.parent == __):
            # Fast path for the common case group().by(key).by(values.().agg())
            # Runs a direct Pandas aggregate over the selected vertices from the
            # last step, then translates the results into the format that
            # a Gremlin group step is supposed to use (key->value pairs)
            values_field = self._value.parent.field_name
            flat_aggs = (
                self.parent.last_vertices()
                    .groupby([self._key])
                    .aggregate({self._key: "first",
                                values_field: self._value.pandas_agg_name()})
            )
            # flat_aggs is a 2-column dataframe of key and value. Convert to
            # a record.
            result_record = {
                r[0]: r[1] for r in flat_aggs.itertuples(index=False)
            }
            self._set_attrs(
                paths=pd.DataFrame({0: [result_record]}),
                step_types=["r"],
                aliases={}
            )

        else:
            # Generic slow path:
            # 1. Build a mapping from grouping key to list of elements
            # 2. Run the subquery on each list
            # 3. Merge the results of the subqueries
            # TODO: Implement this part
            raise NotImplementedError("Slow path of group step not implemented")


class GroupByTraversal(UnaryTraversal):
    """
    Our extended version of the Gremlin `group` step, with easy specification of
    multiple grouping and output columns and the ability to return the
    aggregate as a set of paths, not a single records.

    To use, call the `groupBy()` method of `GraphTraversal`. See that method
    for information about input parameters.
    """
    def __init__(self, parent: GraphTraversal, groups: Sequence[str],
                 aggregates: Sequence[Tuple[str, str, str]]):
        """
        See `GraphTraversal.groupBy()` for parameter info.
        """
        all_agg_aliases = set([a[2] for a in aggregates])
        if len(all_agg_aliases.intersection(groups)) > 0:
            raise ValueError(f"Grouping keys and aggregate names contain "
                             f"duplicate names: "
                             f"{all_agg_aliases.intersection(groups)}")

        UnaryTraversal.__init__(self, parent)
        self._groups = groups
        self._aggregates = aggregates

    def compute_impl(self) -> None:
        # Extract out all the required aliases into a temporary dataframe.
        df_cols = {
            g: self.parent.alias_to_step(g)[1] for g in self._groups
        }
        df_cols.update({
            # Column name in temporary dataframe is output alias, not input,
            # which might not be unique. For example, if we're computing both
            # sum and stdev of the same attribute, each will have its own name.
            out_alias: self.parent.alias_to_step(in_alias)[1].values
            for in_alias, _, out_alias in self._aggregates
        })
        # Build up the arguments to pd.DataFrame.aggregate, starting with
        # grouping columns and then adding aggregates
        agg_args = {g: "first" for g in self._groups}
        agg_args.update({a[2]: a[1] for a in self._aggregates})
        new_paths = (
            pd.DataFrame(df_cols)
            .groupby(self._groups)
            .aggregate(agg_args)
            .reset_index(drop=True)  # No group keys in index
        )
        # Renumber old aliases and add new ones
        new_aliases = {
            self._groups[i]: i for i in range(len(self._groups))
        }
        new_aliases.update({
            # Here we're relying on dict order being guaranteed in Python 3
            # and on the pd.DataFrame constructor respecting that order.
            self._aggregates[i][2]: i + len(self._groups)
            for i in range(len(self._aggregates))
        })
        # Compute the step type metadata for the paths
        new_step_types = [
            self.parent.step_types[self.parent.alias_to_step(g)[0]]
            for g in self._groups
        ] + [
            self.parent.step_types[self.parent.alias_to_step(in_alias)[0]]
            for in_alias, _, _ in self._aggregates
        ]
        self._set_attrs(
            paths=new_paths, aliases=new_aliases, step_types=new_step_types
        )



