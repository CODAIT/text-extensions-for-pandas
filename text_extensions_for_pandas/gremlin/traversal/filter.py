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

# filter.py
#
# Gremlin traversal steps that filter the set of active paths.
import itertools
from typing import Any

from text_extensions_for_pandas import __
from text_extensions_for_pandas.gremlin.predicate import ColumnPredicate, Within
from text_extensions_for_pandas.gremlin.traversal.underscore import \
    find_double_underscore
from text_extensions_for_pandas.gremlin import PrecomputedTraversal

from text_extensions_for_pandas.gremlin.traversal.base import \
    GraphTraversal, UnaryTraversal


class HasTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.has()"""

    def __init__(self, parent: GraphTraversal, key: str, value: Any):
        UnaryTraversal.__init__(self, parent)
        if isinstance(value, ColumnPredicate):
            self._pred = value
        else:
            # Not a predicate ==> Transform to "vertex[key] == value"
            self._pred = Within(value)
        self._pred.modulate(iter(itertools.repeat(key)))

    def compute_impl(self) -> None:
        if len(self.parent.paths.columns) == 0:
            raise ValueError("Cannot call has() on an empty path.")
        vertices_to_check = self.parent.last_vertices()
        if self._pred.target_col not in vertices_to_check.columns:
            # Column not present ==> empty result
            # Can't check this in constructor because vertices might not be
            # initialized.
            filtered_paths = self.parent.paths.iloc[0:0]
        else:
            mask = self._pred(vertices_to_check)
            filtered_paths = self.parent.paths[mask]
        self._set_attrs(paths=filtered_paths)


class WherePredicateTraversal(UnaryTraversal):
    """
    A Gremlin `where` step whose argument is a predicate over the input paths.

    Additional arguments to the predicate, such as field names, may appear `by`
    modulators that come after this step.
    """

    def __init__(self, parent: GraphTraversal, pred: "VertexPredicate"):
        UnaryTraversal.__init__(self, parent)
        self._pred = pred
        self._by_args = []

    def by(self, arg: str):
        """
        `by` modulator for passing in additional arguments to the `where`
        step's predicate.

        :param arg: Argument to add to the `where` step. Gremlin allows single
                    strings, lists of strings, and subqueries that return lists
                    of strings, but only single strings are currently supported
                    here.

        :return: this step, modified in place to have the additional argument.
        """
        if not isinstance(arg, str):
            raise ValueError("The by() modulator to where() currently only "
                             "supports a single string argument. "
                             "Got '{}' of type '{}'."
                             "".format(arg, type(arg)))
        self._by_args.append(arg)
        return self

    def compute_impl(self) -> None:
        # Additional string arguments to the predicate tree come via modulators,
        # so we have to apply them here.
        if len(self._by_args) > 0:
            self._pred.modulate(iter(itertools.cycle(self._by_args)))
        self._pred.bind_aliases(self.parent)
        vertices_to_check = self.parent.last_vertices()
        mask = self._pred(vertices_to_check)
        filtered_paths = self.parent.paths[mask]
        self._set_attrs(paths=filtered_paths)


class WhereSubqueryTraversal(UnaryTraversal):
    """A Gremlin `where` step whose argument is a graph traversal."""

    def __init__(self, parent: GraphTraversal, subquery: GraphTraversal):
        UnaryTraversal.__init__(self, parent)
        self._subquery = subquery

    def compute_impl(self) -> None:
        # Rewrite the subquery to replace "__" with the parent.
        # First we locate the __ and its parent by walking backwards through the
        # subquery's steps
        found_double_underscore, step_after_double_underscore = (
            find_double_underscore(self._subquery)
        )

        # Check for special cases
        if not found_double_underscore:
            # TODO: Implement this case
            raise NotImplementedError("where without __ not implemented")
        elif step_after_double_underscore is None:
            # TODO: Implement this case
            raise NotImplementedError("where(__) not implemented")
        else:
            # Common case: Subquery is a chain of UnaryTraversals starting with
            # a __

            # Create a replacement for __ by adding an artificial leading column
            # to the paths produced by the parent step
            paths_with_leading_col = self.parent.paths.copy()
            paths_with_leading_col.insert(0, "artificial_leading_column",
                                          self.parent.paths.index)
            col_types_with_leading_col = ["a"] + self.parent.step_types
            aliases_with_leading_col = {
                k: v + 1 for k, v in self.parent.aliases.items()
            }
            double_underscore_replacement = (
                PrecomputedTraversal(vertices=self.parent.vertices,
                                     edges=self.parent.edges,
                                     paths=paths_with_leading_col,
                                     step_types=col_types_with_leading_col,
                                     aliases=aliases_with_leading_col))

            step_after_double_underscore.parent = double_underscore_replacement

            # Now we can evaluate the subquery.
            self._subquery.compute()

            # Inspect the output of the subquery and retain every input path for
            # which there is at least one subquery output.
            subquery_paths = self._subquery.paths
            remaining_leading_col_values = subquery_paths[0].unique()

            # Chop off the leading column and generate results
            self._set_attrs(
                paths=(
                    paths_with_leading_col[paths_with_leading_col[0]
                    .isin(remaining_leading_col_values)]
                    .drop("artificial_leading_column", axis="columns"))
            )

            # Reset the subquery so that this step can be recomputed later.
            # TODO: Should this happen in self.uncompute()?
            self._subquery.uncompute()
            step_after_double_underscore.parent = __