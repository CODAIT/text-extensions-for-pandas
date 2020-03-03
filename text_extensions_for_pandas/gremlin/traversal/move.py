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

# move.py
#
# Gremlin traversal steps that move around the graph

from typing import Tuple

from text_extensions_for_pandas.gremlin.traversal.base import \
    GraphTraversal, UnaryTraversal


class OutTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.out()"""

    # TODO: This class ought to be combined with InTraversal, but currently
    #  they are separate as a workaround for some puzzling behavior of pd.merge
    def __init__(self, parent: GraphTraversal, edge_types: Tuple[str]):
        UnaryTraversal.__init__(self, parent)
        self._edge_types = edge_types

    def compute_impl(self) -> None:
        if self.parent.step_types[-1] != "v":
            raise ValueError(
                "Can only call out() when the last element in the path is a "
                "vertex. Last element type is {}".format(
                    self.parent.step_types[-1]))

        # Last column of path is a list of vertices. Join with edges table.
        p = self.parent.paths
        edges = self.parent.edges
        # Filter down to requested edge types if present
        if len(self._edge_types) > 0:
            edges = edges[edges["type"].isin(self._edge_types)]
        edges = edges[["from", "to"]]  # "type" col has served its purpose
        new_paths = (p
                     .merge(edges, left_on=p.columns[-1], right_on="from")
                     .drop("from",
                           axis="columns")  # merge keeps both sides of equijoin
                     # "to" field ==> Last element
                     .rename(columns={"to": len(p.columns)}))
        self._set_attrs(paths=new_paths,
                        step_types=self.parent.step_types + ["v"])


class InTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.in_()"""

    def __init__(self, parent: GraphTraversal, edge_types: Tuple[str]):
        UnaryTraversal.__init__(self, parent)
        self._edge_types = edge_types

    def compute_impl(self) -> None:
        if self.parent.step_types[-1] != "v":
            raise ValueError(
                "Can only call in_() when the last element in the path is a "
                "vertex. Last element type is {}".format(
                    self.parent.step_types[-1]))
        # Last column of path is a list of vertices. Join with edges table.
        merge_tmp = self.parent.paths.copy()
        edges = self.parent.edges
        # Filter down to requested edge types if present
        if len(self._edge_types) > 0:
            edges = edges[edges["type"].isin(self._edge_types)]
        edges = edges[["from", "to"]]  # "type" col has served its purpose
        # pd.merge() doesn't like integer series names for join keys
        merge_tmp["join_key"] = merge_tmp[merge_tmp.columns[-1]]
        new_paths = (
            merge_tmp
            .merge(edges, left_on="join_key", right_on="to")
            .drop(["to", "join_key"], axis="columns")
            .rename(columns={"from": len(self.parent.paths.columns)})
        )
        self._set_attrs(paths=new_paths,
                        step_types=self.parent.step_types + ["v"])