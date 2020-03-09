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

# constant.py
#
# Gremlin traversal steps that always return the same value
from typing import List, Dict, Any

import pandas as pd

from text_extensions_for_pandas.gremlin.traversal.base import \
    BootstrapTraversal, GraphTraversal, UnaryTraversal


class PrecomputedTraversal(BootstrapTraversal):
    """
    Wrapper for the immutable results of a traversal that has already been run.

    Also used for bootstrapping a new traversal.
    """

    def __init__(self, vertices: pd.DataFrame, edges: pd.DataFrame,
                 paths: pd.DataFrame, step_types: List[str],
                 aliases: Dict[str, int]):
        """
        **DO NOT CALL THIS CONSTRUCTOR DIRECTLY.** Use factory methods.

        :param vertices: DataFrame of vertices that make up the graph

        :param edges: DataFrame of edges in the graph, where the "from" and "to"
        columns are references to the index of `vertices`.

        :param paths: DataFrame of paths that this traversal represents. Each
        row represents a path.

        :param step_types: Array of additional information about the columns
        of `paths`. Each value may be None (raw Pandas type), "v" (vertex
        reference), "e" (edge reference), or "r" (record with one or more named
        fields)

        :param aliases: The current set of aliases for path elements, as a map
        from alias name to integer index.
        """
        BootstrapTraversal.__init__(self, vertices, edges)
        self._paths = paths
        self._step_types = step_types
        self._aliases = aliases
        self._is_computed = True

    def compute_impl(self) -> None:
        raise ValueError("This method should never be called")

    @classmethod
    def as_precomputed(cls, t: GraphTraversal, compute: bool = True):
        """
        Factory method for wrapping the output of a traversal in a
        static `PrecomputedTraversal`.

        :param t: A (possibly dynamic) `GraphTraversal`
        :param compute: If True, call the returned object's `compute` method
        before returning it.
        :return: An instance of `PrecomputedTraversal` wrapped around shallow
        copies of the outputs of `t`.
        """
        t.compute()
        ret = PrecomputedTraversal(t.vertices, t.edges, t.paths,
                                   t.step_types, t.aliases)
        if compute:
            ret.compute()
        return ret


class VTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.V()"""

    def __init__(self, parent: GraphTraversal):
        UnaryTraversal.__init__(self, parent)

    def compute_impl(self) -> None:
        if len(self.parent.paths.columns) > 0:
            # TODO: Determine the semantics of calling V() on a non-empty
            #  traversal and implement those semantics here
            raise NotImplementedError("Computing V() on a non-empty traversal "
                                      "not implemented")
        self._set_attrs(paths=pd.DataFrame({0: self.parent.vertices.index}),
                        step_types=["v"], aliases={})


class ConstantTraversal(UnaryTraversal):
    """A Gremlin `constant` step, with some additional information about output
    type that isn't present in the reference implementation of Gremlin."""

    def __init__(self, parent: GraphTraversal, value: Any, step_type: str):
        """
        :param parent: Traversal that produces inputs to this one

        :param value: Constant value that is appended to all paths on the input
        of this traversal.

        :param step_type: Element type string that is appended to the
        `step_types` output of this step. See `step_types` for more
        information.
        """
        UnaryTraversal.__init__(self, parent)
        # TODO: Validate that value and step_type are compatible and raise
        #  an error here instead of downstream if there is a problem.
        self._value = value
        self._step_type = step_type

    def compute_impl(self) -> None:
        self._set_attrs(paths=self._parent_path_plus_elements(self._value),
                        step_types=(self.parent.step_types
                                    + [self._step_type]))
