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

# format.py
#
# Gremlin traversal steps that format and restructure the results of the current
# path.
from typing import Tuple

import pandas as pd

from text_extensions_for_pandas.gremlin.traversal.base import UnaryTraversal, \
    GraphTraversal


class AsTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.as_()"""

    def __init__(self, parent: GraphTraversal, names: Tuple[str]):
        UnaryTraversal.__init__(self, parent)
        self._names = names

    def compute_impl(self) -> None:
        new_aliases = self.parent.aliases.copy()
        for name in self._names:
            new_aliases[name] = len(self.parent.paths.columns) - 1
        self._set_attrs(aliases=new_aliases)


class SelectTraversal(UnaryTraversal):
    """
    A traversal that ends in a `select`, possibly followed by a sequence of
    `by` elements.

    This class contains logic for handling the `by()` arguments to `select()`,
    as well as code for formatting the output of `select()` as a DataFrame.
    """

    def __init__(self, parent: GraphTraversal, selected_aliases: Tuple[str]):
        """
        :param parent: Previous element of the traversal
        :param selected_aliases: List of aliases mentioned as direct arguments
            to `select()`
        """
        UnaryTraversal.__init__(self, parent)
        if 0 == len(selected_aliases):
            raise ValueError("Must select at least one alias")
        self._selected_aliases = selected_aliases
        # List to be populated by self.by()
        self._by_list = []  # Type: List[str]
        # DataFrame or Series representation of the last element of self.paths
        self._paths_tail = None  # Type: Union[pd.DataFrame, pd.Series]

    @property
    def paths(self):
        """
        We override this method of the superclass so that we can defer
        constructing the DataFrame of paths when this step is the last step.

        :return: DataFrame of paths that this traversal represents. Each
        row represents a path; each column represents as step.
        """
        self._check_computed()
        # We cache the result of this method in self._paths.
        if self._paths is None:
            if isinstance(self._paths_tail, pd.DataFrame):
                # Last element is a record.
                # We would like to use a Numpy recarray here, i.e.:
                # last_step = self._paths_tail.to_records(index=False)
                # ...but Pandas does not currently support them as column types.
                # For now, convert each row to a Python dictionary object.
                df = self._paths_tail
                last_step = [{df.columns[i]: t[i]
                              for i in range(len(df.columns))}
                             for t in df.itertuples(index=False)]
            elif isinstance(self._paths_tail, pd.Series):
                last_step = self._paths_tail.values
            else:
                raise ValueError(f"Unexpected type {type(self._paths_tail)}")
            self._paths = self._parent_path_plus_elements(last_step)
        return self._paths

    def compute_impl(self) -> None:
        if len(self._by_list) == 0:
            # No by() modulator ==> Return vertices
            if len(self._selected_aliases) > 1:
                # TODO: Figure out what are the semantics of select(a, b)
                #  without a by() modulator.
                raise NotImplementedError("Multi-alias select without by not "
                                          "implemented")
            _, self._paths_tail = self.parent.alias_to_step(
                self._selected_aliases[0])
            col_type = "v"
        else:
            # by() modulator present ==> Return a column of records
            df_contents = {}
            for i in range(len(self._selected_aliases)):
                alias = self._selected_aliases[i]
                # The Gremlin select statement rotates through the by list if
                # there are more aliases than elements of the by list.
                by = self._by_list[i % len(self._by_list)]
                selected_index = self.parent.aliases[alias]
                if selected_index >= len(self.parent.step_types):
                    raise ValueError(f"Internal error: {alias} points to index "
                                     f"{selected_index}, which is out of range."
                                     f" Aliases table: {self.parent.aliases}\n"
                                     f"Paths table: {self.parent.paths}")
                step_type = self.parent.step_types[selected_index]
                elem = self.parent.paths[selected_index]
                if "v" == step_type:
                    # Vertex element.
                    if by is None:
                        # TODO: Implement this case
                        raise NotImplementedError(
                            "select of entire vertex not implemented")
                    else:
                        if by not in self.parent.vertices.columns:
                            raise ValueError(
                                "Column '{}' not present in vertices of '{}'"
                                "".format(by, alias))
                        df_contents[alias] = self.parent.vertices[by].loc[
                            elem].values
                elif "p" == step_type:
                    if by is not None:
                        # Gremlin requires empty by() for select on scalar step
                        raise ValueError(f"Attempted to apply non-empty by "
                                         f"modulator '{by}' to '{alias}' (step "
                                         f"{selected_index}), which produces "
                                         f"scalar-valued outputs.")
                    df_contents[alias] = elem
                else:
                    raise NotImplementedError(
                        f"select not implemented for step type '{step_type}'")

            self._paths_tail = pd.DataFrame(df_contents)
            col_type = "r"  # Record
        self._set_attrs(
            step_types=self.parent.step_types + [col_type]
        )
        # self._set_attrs() will set self._paths to self.parent.paths
        self._paths = None

    def by(self, field_name: str = None) -> "SelectTraversal":
        """
        `by` modulator to `select()`, for adding a lit of arguments. If the
        number selected aliases exceeds the number of `by` modulators, the
        `select` step will cycle through the list from the beginning.

        Modifies this object in place.

        :param field_name: Next argument in the list of field name arguments to
         select or `None` to select the step's entire output (vertex number or
         literal)

        :returns: A pointer to this object (after modification) to enable
        operation chaining.
        """
        self._by_list.append(field_name)
        return self

    def toList(self):
        """
        Overload the superclass's implementation to generate the results of
        the `select`.
        """
        return self.toDataFrame().to_dict("records")

    def toDataFrame(self):
        """
        :return: A DataFrame with the schema of the `select` statement.
        Column names will be the aliases referenced in the `select()` element,
        and values will be drawn from the vertex/edge columns named in the
        `by` element.
        """
        self.compute()
        last_elem_type = self.step_types[-1]
        if "v" == last_elem_type:
            # Turn the list of vertex IDs into a DataFrame of vertices
            return self.vertices.loc[self._paths_tail.values]
        elif "r" == last_elem_type:
            return self._paths_tail
        else:
            raise ValueError(f"Unexpected last element type '{last_elem_type}'")


class ValuesTraversal(UnaryTraversal):
    """Gremlin `values` step, currently limited to a single field with a single
     value. Interprets `None`/nan as "no value in this field".
     """

    def __init__(self, parent: GraphTraversal, field_name: str):
        UnaryTraversal.__init__(self, parent)
        self._field_name = field_name

    @property
    def field_name(self):
        return self._field_name

    def compute_impl(self) -> None:
        last_step_type = self.parent.step_types[-1]
        if "v" != last_step_type:
            raise NotImplementedError(f"values step over non-vertex element"
                                      f"not implemented (element type "
                                      f"{last_step_type})")
        if self._field_name not in self.parent.vertices.columns:
            # Invalid name --> empty result, to match Gremlin semantics
            new_paths = self.parent.paths.iloc[0:0, :].copy()
            length = len(new_paths.columns)
            new_paths.insert(length, length, [])
            self._set_attrs(paths=new_paths,
                            step_types=self.parent.step_types + ["o"])
            return
        vertex_indexes = self.parent.paths[self.parent.paths.columns[-1]]
        field_values = (self.parent.vertices[self._field_name]
                        .loc[vertex_indexes])
        # TODO: Use a more specific step type than "raw Pandas type"
        step_type = "p"
        self._set_attrs(paths=self._parent_path_plus_elements(field_values),
                        step_types=self.parent.step_types + [step_type])