#
# gremlin.py
#
# Part of pandas_text
#
# Code for running Gremlin queries against parse trees stored as DataFrames.
#
from abc import ABC

import json
import numpy as np
import pandas as pd
import sys
import textwrap
from typing import *


class GraphTraversal:
    """
    Class that represents a [subset of] Gremlin graph traversal.

    Methods that evaluate the traversal operate recursively with memoization,
    and there is no attempt at query optimization.
    This may change in the future.
    """
    def __init__(self):
        # See property getters below for the meanings of these attributes.
        self._vertices = None
        self._edges = None
        self._paths = None
        self._path_col_types = None
        self._aliases = None
        self._is_computed = False

    def _check_computed(self):
        if not self._is_computed:
            # noinspection PyProtectedMember
            raise ValueError("Attempted to call '{}' method on a "
                             "GraphTraversal that had not yet been computed."
                             "".format(sys._getframe(1).f_code.co_name))

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
        row represents a path.
        """
        self._check_computed()
        return self._paths

    @property
    def path_col_types(self):
        """
        :return: Array of additional information about the columns
        of `paths`. Each value may be None (raw Pandas type), "v" (vertex
        reference), "e" (edge reference), or "a" (artifical unique key for
        internal use)
        """
        self._check_computed()
        return self._path_col_types

    @property
    def aliases(self):
        """
        :return: The current set of aliases for path elements, as a map
        from alias name to integer index.
        """
        self._check_computed()
        return self._aliases

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
        col_type = self._path_col_types[-1]
        last_elems = self.paths[self.paths.columns[-1]].tolist()
        if col_type == "v":
            return ["v[{}]".format(elem) for elem in last_elems]
        elif col_type == "e":
            return ["e[{}]".format(elem) for elem in last_elems]
        else:
            return last_elems

    def compute_impl(self) -> None:
        """
        Subclasses should override this method to compute the values of
        `self._vertices` and the other fields that the constructor initializes
        to None.
        """
        raise NotImplementedError("Subclasses need to implement this method")

    def V(self):
        """
        :returns a GraphTraversal that starts a new traversal with all
        vertices of the graph.
        """
        return VTraversal(self)

    def has(self, key: str, value: Any):
        """
        :param key: Key to look for in the properties of the most recent
        vertex or edge

        :param value: Expected value the indicated field in each vertex.
        Can be either a literal value (expected value of a field) or a

        :returns: A GraphTraversal that filters out paths whose last element
        does not contain the indicated value in the indicated field.
        """
        return HasTraversal(self, key, value)

    def as_(self, *names: str):
        """
        :param names: Alias[es] for the current element in the traversal

        :returns: A GraphTraversal that marks the current (rightmost) element
        of this traversal with the indicated short name[s].
        """
        return AsTraversal(self, names)

    def out(self):
        """
        :returns: A GraphTraversal that adds the destination of any edges out
        of the current traversal's last elemehasnt.
        """
        return OutTraversal(self)

    def in_(self):
        """
        :returns: A GraphTraversal that adds the destination of any edges into
        the current traversal's last element.
        """
        return InTraversal(self)

    def select(self, *args):
        """
        :param args: List of 1 or more strings to select

        :returns: A GraphTraversal at the beginning of a `select` operation.
        """
        if 0 == len(args):
            raise ValueError(
                "No arguments passed to select(). Must select at least 1 "
                "alias from the traversal.")
        return SelectTraversal(self, selected_aliases=args, by_list=[])

    def where(self, subquery: "GraphTraversal"):
        """
        The Gremlin `where` step. Performs existential quantification
        roughly equivalent to `WHERE EXISTS (subquery)` in SQL. Usually the
        subquery is correlated and uses the `__` alias for "last step of the
        parent traversal"

        :param subquery: A Gremlin graph traversal, possibly referencing one or
        more parts of the parent traversal via aliases or the special `__`
        built-in alias for "output of the parent step".

        :return: Traversal that returns all input paths for which `subquery`
        produces one or more paths.
        """
        return WhereTraversal(self, subquery)

    def repeat(self, loop_body: "GraphTraversal"):
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
        return RepeatTraversal(self, loop_body)

    def emit(self, emit_pred: "VertexPredicate" = None):
        """
        `emit` modulator: Tells what vertices to emit from a `repeat` step.

        :param emit_pred: Predicate that evaluates to `True` for every vertex
        that should be emitted.

        :return: If this step comes after `repeat()`, a callable traversal that
        performs the `repeat` step according to this modulator. Otherwise
        returns a placeholder that will provide input to subsequent `repeat()`
        calls.
        """
        if emit_pred is None:
            emit_pred = TruePredicate()
        return RepeatTraversal(self, emit_pred=emit_pred)

    def until(self, until_pred: "VertexPredicate"):
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
        return RepeatTraversal(self, until_pred=until_pred)



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
        self._path_col_types = [],
        self._aliases = {}

    def uncompute(self):
        self._paths = None
        self._path_col_types = None
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
        self._path_col_types = None
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
                   path_col_types: List[str] = None,
                   aliases: Dict[str, int] = None):
        """
        Single place for setting computed attributes of subclasses.
        Should only be called from `compute_impl()`.

        Anything set to None gets replaced with the equivalent attribute of
        `self.parent`.
        """
        self._paths = paths if paths is not None else self.parent.paths
        self._path_col_types = (path_col_types if path_col_types is not None
                                else self.parent.path_col_types)
        self._aliases = aliases if aliases is not None else self.parent.aliases

        # Edges and vertices are immutable, but the parent's accessors to
        # them may not be valid until after its' compute method is called.
        self._vertices = self.parent.vertices
        self._edges = self.parent.edges


class PrecomputedTraversal(BootstrapTraversal):
    """
    Wrapper for the immutable results of a traversal that has already been run.

    Also used for bootstrapping a new traversal.
    """
    def __init__(self, vertices: pd.DataFrame, edges: pd.DataFrame,
                 paths: pd.DataFrame, path_col_types: List[str],
                 aliases: Dict[str, int]):
        """
        **DO NOT CALL THIS CONSTRUCTOR DIRECTLY.** Use factory methods.

        :param vertices: DataFrame of vertices that make up the graph

        :param edges: DataFrame of edges in the graph, where the "from" and "to"
        columns are references to the index of `vertices`.

        :param paths: DataFrame of paths that this traversal represents. Each
        row represents a path.

        :param path_col_types: Array of additional information about the columns
        of `paths`. Each value may be None (raw Pandas type), "v" (vertex
        reference), or "e" (edge reference)

        :param aliases: The current set of aliases for path elements, as a map
        from alias name to integer index.
        """
        BootstrapTraversal.__init__(self, vertices, edges)
        self._paths = paths
        self._path_col_types = path_col_types
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
                                   t.path_col_types, t.aliases)
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
                        path_col_types=["v"], aliases={})


class HasTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.has()"""
    def __init__(self, parent: GraphTraversal, key: str, value: Any):
        UnaryTraversal.__init__(self, parent)
        if isinstance(value, ColumnPredicate):
            self._pred = value
        else:
            # Not a predicate ==> Transform to "vertex[key] == value"
            self._pred = Within(value)
        self._pred.target_col = key

    def compute_impl(self) -> None:
        if len(self.parent.paths.columns) == 0:
            raise ValueError("Cannot call has() on an empty path.")
        # Join current path back with vertices table
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


class OutTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.out()"""
    # TODO: This class ought to be combined with InTraversal, but currently
    #  they are separate as a workaround for some puzzling behavior of pd.merge
    def __init__(self, parent: GraphTraversal):
        UnaryTraversal.__init__(self, parent)

    def compute_impl(self) -> None:
        if self.parent.path_col_types[-1] != "v":
            raise ValueError(
                "Can only call out() when the last element in the path is a "
                "vertex. Last element type is {}".format(
                    self.parent.path_col_types[-1]))

        # Column of path is a list of vertices. Join with edges table.
        p = self.parent.paths
        new_paths = (
            p
            .merge(self.parent.edges, left_on=p.columns[-1], right_on="from")
            .drop("from",
                  axis="columns")  # merge keeps both sides of equijoin
            .rename(columns={
                "to": len(p.columns)}))  # "to" field ==> Last element
        self._set_attrs(paths=new_paths,
                        path_col_types=self.parent.path_col_types + ["v"])


class InTraversal(UnaryTraversal):
    """Result of calling GraphTraversal.in_()"""
    def __init__(self, parent: GraphTraversal):
        UnaryTraversal.__init__(self, parent)

    def compute_impl(self) -> None:
        if self.parent.path_col_types[-1] != "v":
            raise ValueError(
                "Can only call in_() when the last element in the path is a "
                "vertex. Last element type is {}".format(
                    self.parent.path_col_types[-1]))
        # Last column of path is a list of vertices. Join with edges table.
        merge_tmp = self.parent.paths.copy()
        # pd.merge() doesn't like integer series names for join keys
        merge_tmp["join_key"] = merge_tmp[merge_tmp.columns[-1]]
        new_paths = (
            merge_tmp
            .merge(self.parent.edges, left_on="join_key", right_on="to")
            .drop(["to", "join_key"], axis="columns")
            .rename(columns={"from": len(self.parent.paths.columns)})
        )
        self._set_attrs(paths=new_paths,
                        path_col_types=self.parent.path_col_types + ["v"])


class SelectTraversal(UnaryTraversal):
    """
    A traversal that ends in a `select`, possibly followed by a sequence of
    `by` elements.

    This class contains logic for handling the `by()` arguments to `select()`,
    as well as code for formatting the output of `select()` as a DataFrame.
    """

    def __init__(self, parent: GraphTraversal, selected_aliases: Tuple[str],
                 by_list: List[str]):
        """
        :param parent: Previous element of the traversal
        :param selected_aliases: List of aliases mentioned as direct arguments
            to `select()`
        :param by_list: List of field names passed to this `select` so far
        through `by` elements.
        """
        UnaryTraversal.__init__(self, parent)
        self._selected_aliases = selected_aliases
        self._by_list = by_list

    def compute_impl(self) -> None:
        # This step only formats the results of previous steps, so no
        # computation beyond the base class's calling of self.parent.compute()
        # is necessary.
        self._set_attrs()

    def by(self, field_name: str):
        """
        :field_name: Next argument in the list of field name arguments to select
        """
        return SelectTraversal(self, self._selected_aliases,
                               self._by_list + [field_name])

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
        # Schema of a Gremlin select is: (key1, value1), (key2, value2), ...
        # where key1, key2, ... are the aliases and value1, value2, ... are the
        # extracted field values.
        if 0 == len(self._by_list):
            # TODO: Implement this branch
            raise NotImplementedError("select without by not yet implemented")
        self.compute()
        df_contents = {}
        for i in range(len(self._selected_aliases)):
            alias = self._selected_aliases[i]
            # The Gremlin select statement rotates through the by list if there
            # are more aliases than elements of the by list.
            by = self._by_list[i % len(self._by_list)]
            selected_index = self.aliases[alias]  # index into traversal
            col_type = self.path_col_types[selected_index]
            elem = self.paths[selected_index]
            if "v" == col_type:
                # Vertex element.
                if by not in self.parent.vertices.columns:
                    raise ValueError(
                        "Column '{}' not present in vertices of '{}'"
                        "".format(by, alias))
                df_contents[alias] = self.parent.vertices[by].loc[elem].values
            else:
                # TODO: Implement this branch
                raise NotImplementedError(
                    "select on non-vertex element not implemented")
        return pd.DataFrame(df_contents)


class DoubleUnderscore(GraphTraversal):
    """
    Standin for Gremlin's `__` (two underscore characters) operation, which
    means "anonymous traversal starting at the end of the current traversal".

    This object doesn't itself perform any processing itself. Upstream rewrites
    replace these placeholders with the appropriate concrete subqueries.
    """
    def _not_impl(self):
        raise NotImplementedError("This object is a placeholder whose methods "
                                  "should never be called. Instead, rewrites "
                                  "should replace this object with a callable "
                                  "instance of GraphTraversal.")

    @property
    def edges(self):
        self._not_impl()

    @property
    def vertices(self):
        self._not_impl()

    def compute(self) -> None:
        self._not_impl()

    def compute_impl(self) -> None:
        self._not_impl()

    def uncompute(self):
        pass


# Alias to allow "pt.__" in Gremlin expressions
__ = DoubleUnderscore()


def find_double_underscore(last_step: GraphTraversal) -> Tuple[bool,
                                                               GraphTraversal]:
    """
    Common subroutine of steps that need to find the "__" in their arguments.
    :param last_step: Last step of a sub-traversal.
    :return: A tuple consisting of:
    * a boolean value that is True if `__` was found at the beginning of the
      sub-traversal.
    * The node immediately after the `__`
    """
    cur_step = last_step
    step_after_cur_step = None
    while True:
        if isinstance(cur_step, (PrecomputedTraversal, BootstrapTraversal)):
            # Reached a raw input without finding a __
            found_double_underscore = False
            break
        elif cur_step == __:
            found_double_underscore = True
            break
        elif isinstance(cur_step, UnaryTraversal):
            step_after_cur_step = cur_step
            cur_step = cur_step.parent
        else:
            raise ValueError("Don't know how to rewrite an instance of "
                             "'{}'".format(type(cur_step)))
    return found_double_underscore, step_after_cur_step


class WhereTraversal(UnaryTraversal):
    """A Gremlin `where` step."""
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
            col_types_with_leading_col = ["a"] + self.parent.path_col_types
            double_underscore_replacement = (
                PrecomputedTraversal(vertices=self.parent.vertices,
                                     edges=self.parent.edges,
                                     paths=paths_with_leading_col,
                                     path_col_types=col_types_with_leading_col,
                                     aliases=self.parent.aliases))

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


class RepeatTraversal(UnaryTraversal):
    # TODO: Determine semantics of multiple emit modulators
    # TODO: Determine semantics of ...repeat().emit().repeat()...
    # TODO: Determine semantics of ...repeat().until().repeat()...
    """
    A Gremlin `repeat` step, possibly modified by `emit` and/or `until`.
    Can also represent an `emit` or `until` modulator that occurs before the
    `repeat`step that it modifies.
    """
    def __init__(self, parent: GraphTraversal,
                 loop_body: GraphTraversal = None,
                 emit_pred: "VertexPredicate" = None,
                 until_pred: "VertexPredicate" = None
                 ):
        """
        This initializer is called from `repeat()`, `emit()`, and `until()`.
        Note that `emit` and `until` modulators can come before `repeat`.

        :param parent: Previous step in the traversal
        :param loop_body: Argument to the `repeat` step if called from the
        `repeat` step itself; otherwise `None`.
        :param emit_pred: Argument to the `emit` modulator, if present;
        otherwise None.
        :param until_pred: Argument to the `until` modulator, if present;
        otherwise None.
        """
        # Look at earlier steps for information about the pieces of the repeat
        # step.
        prev_loop_body = prev_emit_pred = prev_until_pred = None
        non_repeat_parent = parent  # Last step before entire repeat "clause"
        if isinstance(parent, RepeatTraversal):
            prev_loop_body = parent._loop_body
            prev_emit_pred = parent._init_emit_pred
            prev_until_pred = parent._init_until_pred
            while isinstance(non_repeat_parent, RepeatTraversal):
                non_repeat_parent = non_repeat_parent.parent

        # Note that we initialize the parent pointer to the step before the
        # entire repeat "clause".
        UnaryTraversal.__init__(self, non_repeat_parent)

        self._loop_body =\
            self._non_null_value(loop_body, prev_loop_body, "loop body")
        self._init_emit_pred = \
            self._non_null_value(emit_pred, prev_emit_pred, "emit predicate")
        self._init_until_pred = \
            self._non_null_value(until_pred, prev_until_pred, "until predicate")

        # Convert None in any of the predicate fields to default value for the
        # purposes of self.compute()
        self._emit_pred = (self._init_emit_pred
                           if self._init_emit_pred is not None
                           else FalsePredicate())
        self._until_pred = (self._until_pred
                            if self._init_until_pred is not None
                            else TruePredicate())

        # The semantics of repeat() are different depending on whether the emit
        # and until modulators come before or after.
        if loop_body is None and prev_loop_body is None:
            # No loop body at this point or before.
            self._emit_before_repeat = None
            self._until_before_repeat = None
        elif loop_body is None and prev_loop_body is not None:
            # Modulator that comes comes after the repeat step
            self._emit_before_repeat = parent._emit_before_repeat
            self._until_before_repeat = parent._until_before_repeat
        elif loop_body is not None:
            # The repeat step itself
            self._emit_before_repeat = (prev_emit_pred is not None)
            self._until_before_repeat = (prev_until_pred is not None)
        else:
            raise ValueError("This code should be unreachable")

    @staticmethod
    def _non_null_value(a, b, description):
        """Return whichever of a and b is not None and raise an error if
        both are not None."""
        if a is None and b is None:
            return None
        elif a is not None and b is not None:
            raise ValueError("Got two conflicting values for {}: {} and {}"
                             "".format(description, a, b))
        elif a is not None and b is None:
            return a
        elif b is not None and a is None:
            return b
        else:
            raise ValueError("This code should be unreachable ({}, {})"
                             "".format(a, b))

    def _add_to_emit_list(self, t: GraphTraversal,
                          emit_list: List[pd.DataFrame],
                          emit_type: str) -> str:
        """
        Extract out the current iteration's paths to emit.
        :param t: Tail of the curent iteration of the loop
        :param emit_list: List of DataFrame fragments containing the emitted
        paths from previous iterations.
        :param emit_type: Type returned

        :returns: The next value to be passed in for emit_type
        """
        # emit_pred tells which rows to retain
        mask = self._emit_pred(t.last_step())
        full_paths = t.paths[mask]
        next_emit_type = t.path_col_types[-1]
        if emit_type is not None and emit_type != next_emit_type:
            raise ValueError("Different iterations of repeat() would emit "
                             "different types: {} and {}"
                             "".format(emit_type, next_emit_type))
        # Retain the path elements provided by the parent, plus the last
        # element of each path.
        num_parent_steps = len(self.parent.paths.columns)
        df = full_paths[full_paths.columns[0:num_parent_steps]]
        df.insert(loc=num_parent_steps, column=num_parent_steps,
                  value=full_paths[full_paths.columns[-1]])
        emit_list.append(df)
        return next_emit_type

    def compute_impl(self) -> None:
        # We will rewrite the loop body repeatedly, replacing double underscore
        # with the results of the previous iteration each time. Locate that
        # double underscore step.
        found_double_underscore, step_after_double_underscore = (
            find_double_underscore(self._loop_body)
        )
        # Check for special cases
        if not found_double_underscore:
            # TODO: Implement this case
            raise NotImplementedError("repeat without __ not implemented")
        elif step_after_double_underscore is None:
            # TODO: Implement this case
            raise NotImplementedError("repeat(__) not implemented")
        else:
            # Common case: Loop body is a chain of UnaryTraversals starting with
            # a __.
            if (self._until_before_repeat and
                    np.any(self._until_pred(self.parent.last_vertices()))):
                # until() before repeat(), which means "do-while" semantics,
                # and until predicate fired before first iteration.
                self._set_attrs()
                return

            emit_list = []  # type: List[pd.DataFrame]
            emit_type = None

            # emit() before repeat() --> emit predicate applied before the first
            # iteration, using the parent's output as input.
            # TODO: Determine whether we should skip applying the emit predicate
            #  after the last iteration in this case.
            if self._emit_before_repeat:
                emit_type = self._add_to_emit_list(self.parent, emit_list,
                                                   emit_type)

            # First iteration is special.
            iteration_counter = 0
            step_after_double_underscore.parent = (
                PrecomputedTraversal.as_precomputed(self.parent))
            self._loop_body.compute()
            emit_type = self._add_to_emit_list(self._loop_body, emit_list,
                                               emit_type)
            prev_iter_output = (
                PrecomputedTraversal.as_precomputed(self._loop_body))
            self._loop_body.uncompute()

            # Iterations 2 and onward
            while not np.any(
                    self._until_pred(prev_iter_output.last_vertices())):
                iteration_counter += 1
                step_after_double_underscore.parent = prev_iter_output
                self._loop_body.compute()
                emit_type = self._add_to_emit_list(self._loop_body, emit_list,
                                                   emit_type)
                prev_iter_output = (
                    PrecomputedTraversal.as_precomputed(self._loop_body))
                self._loop_body.uncompute()

            # Fully reset the loop body so that this step can be recomputed
            # later.
            step_after_double_underscore.parent = __

            # Output all the paths that passed the emit predicate
            new_paths = pd.concat(emit_list)
            new_path_col_types = (
                self.parent.path_col_types
                + [prev_iter_output.path_col_types[-1]])
            # TODO: Raise NotImplementedError if different iterations emitted
            #  different types
            self._set_attrs(paths=prev_iter_output.paths,
                            path_col_types=new_path_col_types)


class VertexPredicate:
    """
    Base class for Boolean predicates applied to individual vertices of the
    graph.
    """
    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        """
        :param vertices: DataFrame of vertices on which to apply the predicate
        :return: A numpy Boolean mask containing `True` for each row that
        satisfies the predicate
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ColumnPredicate(VertexPredicate, ABC):
    """
    Abstract base class for VertexPredicates that only read one column and may
    need that column to be bound late, as in a `has` step.
    """
    def __init__(self):
        self._target_col = None

    @property
    def target_col(self) -> str:
        """
        :returns: Name of the column on whih this predicate will be applied.
        """
        return self._target_col

    @target_col.setter
    def target_col(self, target_col):
        self._target_col = target_col
        self._propagate_target_col()

    def _propagate_target_col(self) -> None:
        """
        Subclasses that represent nested expressions should override this method
        to pass down the value of `self._target_col` any subexpressions.
        """
        pass


class TruePredicate(ColumnPredicate):
    """
    Predicate that always returns `True`
    """
    def __call__(self, values: pd.DataFrame) -> np.ndarray:
        return np.full_like(values.index, True, dtype=np.bool)


class FalsePredicate(ColumnPredicate):
    """
    Predicate that always returns `False`
    """
    def __call__(self, values: pd.DataFrame) -> np.ndarray:
        return np.full_like(values.index, False, dtype=np.bool)


class Within(ColumnPredicate):
    """
    Implementation of the Gremlin `within()` predicate
    """
    def __init__(self, *args: Any):
        """
        :param args: 1 or more arguments to the predicate as Python varargs.
        Currently the class passes these objects through to Pandas' comparison
        functions, but future versions may perform some additional validation
        here.
        :param field
        """
        self._args = args

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        return vertices[self.target_col].isin(self._args).values


class Without(ColumnPredicate):
    """
    Implementation of the Gremlin `without()` predicate
    """
    def __init__(self, *args: Any):
        """
        :param args: 1 or more arguments to the predicate as Python varargs.
        Currently the class passes these objects through to Pandas' comparison
        functions, but future versions may perform some additional validation
        here.
        """
        self._args = args

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        return (~vertices[self.target_col].isin(self._args)).values


def token_features_to_traversal(token_features: pd.DataFrame,
                                drop_self_links=True):
    """
    Turn a DataFrame of token features in the form returned by
    `make_tokens_and_features` into an empty graph traversal.
    Similar to calling `graph.traversal()` in Gremlin.

    :param token_features: DataFrame containing information about individual
    tokens. Must contain a `head_token_num` column that is synchronized with
    the DataFrame's index.

    :param drop_self_links: If `True`, remove links from nodes to themselves
    to simplify query logic.

    :returns: A traversal containing a graph version of `token_features` and
    an empty set of paths.
    """
    # Don't include token IDs in the vertex attributes
    vertices = token_features.drop(["token_num", "head_token_num"], axis=1)
    edges = pd.DataFrame(
        {"from": token_features.index, "to": token_features["head_token_num"]})
    if drop_self_links:
        edges = edges[edges["from"] != edges["to"]]
    paths = pd.DataFrame()
    path_col_types = []
    aliases = {}
    return PrecomputedTraversal(vertices, edges, paths, path_col_types, aliases)


def token_features_to_gremlin(token_features: pd.DataFrame,
                              drop_self_links=True):
    """
    :param token_features: A subset of a token features DataFrame in the format
    returned by `make_tokens_and_features()`.

    :param drop_self_links: If `True`, remove links from nodes to themselves
    to simplify query logic.

    :return: A string of Gremlin commands that you can paste into the Gremlin
    console to generate a graph that models the contents of `token_features`.
    """
    def _quote_str(v):
        return json.dumps(str(v))

    # Nodes:
    # For each token, generate addV("token").property("key","value")...as(id)
    node_lines = []
    colnames = token_features.columns
    for row in token_features.itertuples(index=True):
        # First element in tuple is index value
        index_val = row[0]
        props_list = [".property({}, {})".format(
            _quote_str(colnames[i]), _quote_str(row[i + 1]))
            for i in range(len(colnames))]
        props_str = "".join(props_list)
        node_lines.append("""addV("token"){}.as({})""".format(
            props_str, _quote_str(index_val)))

    # Edges:
    # For each token, generate addE("head").from(token_id).to(head_id)
    edge_lines = []
    for index, value in token_features["head_token_num"].items():
        if drop_self_links and index == value:
            continue
        edge_lines.append("""addE("head").from({}).to({})""".format(
            _quote_str(index), _quote_str(value)))

    # Combine insertions into a single Gremlin statement
    result = """
    g = TinkerGraph.open().traversal()
    g.
    {}.
    {}.
    iterate()
    """.format(
        ".\n    ".join(node_lines),
        ".\n    ".join(edge_lines), )
    return textwrap.dedent(result)
