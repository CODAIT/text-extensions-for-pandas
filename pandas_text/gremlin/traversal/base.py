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
import sys
from abc import ABC
from typing import *

import pandas as pd

# base.py
#
# Abstract base classes for Gremlin traversal steps


class GraphTraversalBase(ABC):
    """
    Base class containing methods we want hidden from the public
    `GraphTraversal` class.
    """

    def __init__(self):
        # See property getters below for the meanings of these attributes.
        self._vertices = None
        self._edges = None
        self._paths = None
        self._step_types = None
        self._aliases = None
        self._is_computed = False

    def _check_computed(self):
        if not self._is_computed:
            # noinspection PyProtectedMember
            raise ValueError("Attempted to call '{}' method on a "
                             "GraphTraversal that had not yet been computed."
                             "".format(sys._getframe(1).f_code.co_name))

    def compute_impl(self) -> None:
        """
        Subclasses should override this method to compute the values of
        `self._vertices` and the other fields that the constructor initializes
        to None.
        """
        raise NotImplementedError("Subclasses need to implement this method")


class GraphTraversal(GraphTraversalBase, ABC):
    """
    Main API entry point for Gremlin traversals.

    Class that represents a [subset of] Gremlin graph traversal.

    Methods that evaluate the traversal operate recursively with memoization,
    and there is no attempt at query optimization.
    This may change in the future.
    """
    @property
    def vertices(self):
        """
        :return: DataFrame of vertices that make up the graph
        """
        self._check_computed()
        return self._vertices

    @property
    def edges(self):
        """
        :return: DataFrame of edges in the graph, where the "from" and "to"
        columns are references to the index of `vertices`.
        """
        self._check_computed()
        return self._edges

    @property
    def paths(self):
        """
        :return: DataFrame of paths that this traversal represents. Each
        row represents a path; each column represents as step.
        """
        self._check_computed()
        return self._paths

    @property
    def step_types(self):
        """
        :return: Array of additional information about the columns
        of `paths`. Each value may be one of:
         * "p" (raw Pandas type)
         * "r" (record with 1 or more fields)
         * "v" (vertex reference)
         * "e" (edge reference)
         * "a" (artificial unique key for internal use)
        """
        self._check_computed()
        return self._step_types

    @property
    def aliases(self) -> Dict[str, int]:
        """
        :return: The current set of aliases for path elements, as a map
        from alias name to integer index.
        """
        self._check_computed()
        return self._aliases

    def alias_to_step(self, alias: str) -> Tuple[int, pd.Series]:
        """
        Convenience method that uses the aliases table to extract a reference
        to the vertices or edges returned by a step.

        :param alias: Key to this traversal's aliases table. Must point to a
        vertex step.

        :return: A tuple consisting of:
            * The step number that the alias points to
            * The vertices or edges that comprise the output of the step,
              as a `pd.Series`
        """
        self._check_computed()
        step_num = self.aliases[alias]
        if step_num is None:
            raise ValueError("Alias '{}' not found (valid aliases are {})"
                             "".format(alias, self.aliases.keys()))
        step_series = self.paths[self.paths.columns[step_num]]
        return step_num, step_series

    def alias_to_vertices(self, alias: str) -> pd.DataFrame:
        """
        Convenience method that uses the aliases table to extract the vertices
        returned by a step. Also makes sure they are actually vertices.

        :param alias: Key to this traversal's aliases table. Must point to a
        vertex step.

        :return: The vertices that comprise the output of the step, formatted
        as a `pd.DataFrame`.
        """
        self._check_computed()
        step_num, step_series = self.alias_to_step(alias)
        if self.step_types[step_num] != "v":
            raise ValueError("Alias '{}' points to a step of type '{}'; should "
                             "be 'v' (for 'vertex')"
                             "".format(alias, self.step_types[step_num]))
        return self.vertices.loc[step_series]

    def last_step(self) -> pd.Series:
        """
        Convenience method to extract the rightmost element of `self.paths`

        :return: the last (rightmost) step of all paths for this
         traversal.
        """
        return self.paths[self.paths.columns[-1]]

    def last_vertices(self) -> pd.DataFrame:
        """
        Convenience method that uses the rightmost element of `self.paths` to
        filter `self.vertices`

        :return: The vertices of the last (rightmost) step of all paths for this
         traversal.
        """
        return self.vertices.loc[self.paths[self.paths.columns[-1]]]

    def compute(self) -> "GraphTraversal":
        """
        Idempotent method to compute and materialize the results of the
        traversal, which can then be accessed via this object's properties.

        :returns: This step, to enable method chaining
        """
        raise NotImplementedError("Subclasses need to implement this method")

    def uncompute(self):
        """
        Idempotent method to recursively remove the results of any previous
        calls to `compute` and mark this step as not computed.
        """
        raise NotImplementedError("Subclasses need to implement this method")

    def toList(self):
        """
        :returns the rightmost element of the traversal as a Python list
        """
        self.compute()
        col_type = self._step_types[-1]
        last_elems = self.paths[self.paths.columns[-1]].tolist()
        if col_type == "v":
            return ["v[{}]".format(elem) for elem in last_elems]
        elif col_type == "e":
            return ["e[{}]".format(elem) for elem in last_elems]
        else:
            return last_elems

    def V(self):
        """
        :returns a GraphTraversal that starts a new traversal with all
        vertices of the graph.
        """
        # local import here and throughout this file because the Gremlin APIs
        # we implement have implicit circular dependencies.
        from pandas_text.gremlin.traversal.constant import VTraversal
        return VTraversal(self)

    def has(self, key: str, value: Any) -> "GraphTraversal":
        """
        :param key: Key to look for in the properties of the most recent
        vertex or edge

        :param value: Expected value the indicated field in each vertex.
        Can be either a literal value (expected value of a field) or a

        :returns: A GraphTraversal that filters out paths whose last element
        does not contain the indicated value in the indicated field.
        """
        from pandas_text.gremlin.traversal.filter import HasTraversal
        return HasTraversal(self, key, value)

    def as_(self, *names: str) -> "GraphTraversal":
        """
        :param names: Alias[es] for the current element in the traversal

        :returns: A GraphTraversal that marks the current (rightmost) element
        of this traversal with the indicated short name[s].
        """
        from pandas_text.gremlin.traversal.format import AsTraversal
        return AsTraversal(self, names)

    def out(self, *edge_types: str) -> "GraphTraversal":
        """
        :param edge_types: 0 or more names of types of edges.
         Zero types means "all edge types".

        :returns: A GraphTraversal that adds the destination of any edges out
        of the current traversal's last elemehasnt.
        """
        from pandas_text.gremlin.traversal.move import OutTraversal
        return OutTraversal(self, edge_types)

    def in_(self, *edge_types: str) -> "GraphTraversal":
        """
        :param edge_types: 0 or more names of types of edges.
         Zero types means "all edge types".

        :returns: A GraphTraversal that adds the destination of any edges into
        the current traversal's last element.
        """
        from pandas_text.gremlin.traversal.move import InTraversal
        return InTraversal(self, edge_types)

    def select(self, *args: str) -> "GraphTraversal":
        """
        :param args: List of 1 or more strings to select

        :returns: A GraphTraversal at the beginning of a `select` operation.
        """
        if 0 == len(args):
            raise ValueError(
                "No arguments passed to select(). Must select at least 1 "
                "alias from the traversal.")
        from pandas_text.gremlin.traversal.format import SelectTraversal
        return SelectTraversal(self, selected_aliases=args)

    def where(self, target: Union["GraphTraversal",
                                  "VertexPredicate"]) -> "GraphTraversal":
        """
        The Gremlin `where` step. Performs existential quantification
        roughly equivalent to `WHERE EXISTS (subquery)` in SQL. Usually the
        subquery is correlated and uses the `__` alias for "last step of the
        parent traversal"

        :param target: One of:
        * A Gremlin graph traversal, possibly referencing one or
          more parts of the parent traversal via aliases or the special `__`
          built-in alias for "output of the parent step".
        * A filtering predicate to be applied to the paths that enter this step.

        :return: Traversal that returns all input paths for which `subquery`
        produces one or more paths.
        """
        from pandas_text.gremlin.traversal.filter import \
            WhereSubqueryTraversal, WherePredicateTraversal
        from pandas_text.gremlin.predicate import VertexPredicate
        if isinstance(target, GraphTraversal):
            return WhereSubqueryTraversal(self, target)
        elif isinstance(target, VertexPredicate):
            return WherePredicateTraversal(self, target)
        else:
            raise ValueError("Unexpected type '{}' of argument to where"
                             "".format(type(target)))

    def repeat(self, loop_body: "GraphTraversal") -> "GraphTraversal":
        """
        `repeat` step: Repeat `loop_body` until the predicate in the associated
        `until` modulator filters
        out all paths output by the previous step, emitting anything that passes
        the predicate in the `emit` clause.

        Can also be modified by the `emit` and `until` modulators. For extra
        fun, `emit` and/or `until` can come *before* the `repeat` step, in which
        case the `repeat` step will have "do-while" semantics.

        If there is no `until` modulator, then the default until predicate is
        `TruePredicate`, meaning, "keep going until an iteration
        does not return any paths".

        If there is no `emit` modulator, then a default emit predicate is
        `FalsePredicate`, meaning "don't emit any elements".

        :param loop_body: Sub-traversal to be repeated

        :return: Traversal that returns the result of repeating `loop_body` the
        appropriate number of times, as modified by any surrounding modulators.
        """
        from pandas_text.gremlin.traversal.recurse import RepeatTraversal
        return RepeatTraversal(self, loop_body)

    def emit(self, emit_pred: "VertexPredicate" = None) -> "GraphTraversal":
        """
        `emit` modulator: Tells what vertices to emit from a `repeat` step.

        :param emit_pred: Predicate that evaluates to `True` for every vertex
        that should be emitted.

        :return: If this step comes after `repeat()`, a callable traversal that
        performs the `repeat` step according to this modulator. Otherwise
        returns a placeholder that will provide input to subsequent `repeat()`
        calls.
        """
        from pandas_text.gremlin.predicate import TruePredicate
        if emit_pred is None:
            emit_pred = TruePredicate()
        from pandas_text.gremlin.traversal.recurse import RepeatTraversal
        return RepeatTraversal(self, emit_pred=emit_pred)

    def until(self, until_pred: "VertexPredicate") -> "GraphTraversal":
        """
        `until` modulator: Tells when to stop a `repeat` step.

        :param until_pred: Predicate to apply to every path that the repeat step
         produces. If the predicate evaluates to `True` for any row, the
         `repeat` step stops.

        :return:If this step comes after `repeat()`, a callable traversal that
        performs the `repeat` step according to this modulator. Otherwise
        returns a placeholder that will provide input to subsequent `repeat()`
        calls.
        """
        from pandas_text.gremlin.traversal.recurse import RepeatTraversal
        return RepeatTraversal(self, until_pred=until_pred)

    def constant(self, value: Any,
                 step_type: str = "p") -> "GraphTraversal":
        """
        `constant` step. Adds the indicated constant value to each path in the
        parent traversal.

        :param value: Value to append. Can be `None`.
        :param step_type: Optional scalar step type string. See
        `GraphTraversal.step_types` for possible values.

        :returns: A GraphTraversal that adds the indicated `constant` step to
        the parent traversal.
        """
        from pandas_text.gremlin.traversal.constant import ConstantTraversal
        return ConstantTraversal(self, value, step_type)

    def coalesce(self, *subqueries: "GraphTraversal") -> "GraphTraversal":
        """
        A Gremlin `coalesce` step. For each path emitted by the parent
         traversal, executes the traversals in `subqueries` in order until
         it finds one that returns at least one result.

        :param subqueries: Sub-traversals to run, in the order that they should
         be tried.

        :return: A GraphTraversal that adds the indicated `coalesce` step to
        the parent traversal.
        """
        from pandas_text.gremlin.traversal.recurse import CoalesceTraversal
        return CoalesceTraversal(self, subqueries)

    def values(self, field_name: str) -> "GraphTraversal":
        """
        A Gremlin `values` step. Expects the last element of the current path to
         be a vertex/edge reference. Adds to the end of each path the
         values of the indicated field, or removes the current path if the
         field either is not present or contains `None`/nil.

        :param field_name: Name of the field to retrieve
        :return: A GraphTraversal that adds the indicated `values` step to
        the parent traversal.
        """
        from pandas_text.gremlin.traversal.format import ValuesTraversal
        return ValuesTraversal(self, field_name)

    def sum(self) -> "GraphTraversal":
        """
        A Gremlin `sum` step. Expects the last element of the current path to be
        a scalar value. Returns the scalar sum as a single path containing one
        element.

        :return: A GraphTraversal that adds the indicated `sum` step to
        the parent traversal.
        """
        from pandas_text.gremlin.traversal.aggregate import SumTraversal
        return SumTraversal(self)


class BootstrapTraversal(GraphTraversal):
    """
    A traversal that has no inputs but does have a graph.
    """

    def __init__(self, vertices: pd.DataFrame, edges: pd.DataFrame):
        """
        Initialize the common attributes of subclasses.

        :param vertices: DataFrame of vertices that make up the graph

        :param edges: DataFrame of edges in the graph, where the "from" and "to"
        columns are references to the index of `vertices`.
        """
        GraphTraversal.__init__(self)
        self._vertices = vertices
        self._edges = edges

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    def compute(self) -> GraphTraversal:
        if not self._is_computed:
            self.compute_impl()
            self._is_computed = True
        return self

    def compute_impl(self) -> None:
        self._paths = pd.DataFrame(),
        self._step_types = [],
        self._aliases = {}

    def uncompute(self):
        self._paths = None
        self._step_types = None
        self._is_computed = False


class UnaryTraversal(GraphTraversal, ABC):
    """
    Abstract base class for traversals that have one input and derive their
    nodes and edges from that input.

    Also takes care of calling the parent's `compute()` method before entering
    `compute_impl()`, as well as propagating the parent's vertices and edges.
    """

    def __init__(self, parent: GraphTraversal):
        GraphTraversal.__init__(self)
        self._parent = parent
        self._vertices = None
        self._edges = None

    def compute(self) -> GraphTraversal:
        if not self._is_computed:
            self.parent.compute()
            self.compute_impl()
            self._is_computed = True
        return self

    def uncompute(self):
        self._vertices = None
        self._edges = None
        self._paths = None
        self._step_types = None
        self._aliases = None
        self._is_computed = False
        if self.parent._is_computed:
            self.parent.uncompute()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent) -> None:
        """
        This method is for use by query rewrite code.

        :param new_parent: New parent pointer for this step of the traversal
        """
        if self._is_computed:
            raise ValueError("Attempted to update parent of {} after "
                             "`compute()` was called.".format(self))
        self._parent = new_parent

    def _set_attrs(self, paths: pd.DataFrame = None,
                   step_types: List[str] = None,
                   aliases: Dict[str, int] = None):
        """
        Single place for setting computed attributes of subclasses.
        Should only be called from `compute_impl()`.

        Anything set to None gets replaced with the equivalent attribute of
        `self.parent`.

        This method also takes care of ensuring that the column names of `paths`
        are a contiguous range of numbers.
        """
        if paths is None:
            self._paths = self.parent.paths
        else:
            # Ensure that the column names are a contiguous range of ints
            paths.columns = list(range(len(paths.columns)))
            self._paths = paths
        self._step_types = (step_types if step_types is not None
                            else self.parent.step_types)
        self._aliases = aliases if aliases is not None else self.parent.aliases

        # Edges and vertices are immutable, but the parent's accessors to
        # them may not be valid until after its' compute method is called.
        self._vertices = self.parent.vertices
        self._edges = self.parent.edges

    def _parent_path_plus_elements(self, next_elements: Any) -> pd.DataFrame:
        """
        :param next_elements: Object containing new elements to tack onto the
        parent path. Must be of a type that Pandas knows how to convert to a
        `pd.Series`.

        :return: A copy of `self.parent.paths` with `next_elements` appended
        to the end.
        """
        new_paths = self.parent.paths.copy()
        # We don't use the index of the paths dataframe, but gaps can make
        # Pandas operations fail. So always reset indexes.
        new_paths = new_paths.reset_index(drop=True)
        if isinstance(next_elements, pd.Series):
            next_elements = next_elements.reset_index(drop=True)
        new_path_position = len(new_paths.columns)
        new_paths.insert(loc=new_path_position, column=new_path_position,
                         value=next_elements)  # Modifies new_paths in place
        return new_paths



