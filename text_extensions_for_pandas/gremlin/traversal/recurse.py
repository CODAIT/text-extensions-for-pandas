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

# recurse.py
#
# Gremlin traversal steps that recursively invoke subqueries.
from typing import List, Sequence

import numpy as np
import pandas as pd

from text_extensions_for_pandas.gremlin.predicate import VertexPredicate, \
    TruePredicate, FalsePredicate
from text_extensions_for_pandas.gremlin.traversal.base import UnaryTraversal, \
    GraphTraversal
from text_extensions_for_pandas.gremlin.traversal.constant import \
    PrecomputedTraversal
from text_extensions_for_pandas.gremlin.traversal.underscore import \
    find_double_underscore, __


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
                 emit_pred: VertexPredicate = None,
                 until_pred: VertexPredicate = None
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

        self._loop_body = \
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
                            else FalsePredicate())

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
        next_emit_type = t.step_types[-1]
        if emit_type is not None and emit_type != next_emit_type:
            raise ValueError(f"Different iterations of repeat() would emit "
                             f"different types: {emit_type} and "
                             f"{next_emit_type}")
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

            # For the first iteration, we substitute __ with the parent step.
            iteration_counter = 0
            step_after_double_underscore.parent = (
                PrecomputedTraversal.as_precomputed(self.parent))
            self._loop_body.compute()
            emit_type = self._add_to_emit_list(self._loop_body, emit_list,
                                               emit_type)
            prev_iter_output = (
                PrecomputedTraversal.as_precomputed(self._loop_body))
            self._loop_body.uncompute()

            until_mask = self._until_pred(prev_iter_output.last_vertices())

            # Iterations 2 and onward
            # np.any() returns false when fed empty input
            while len(until_mask) > 0 and not np.any(until_mask):
                iteration_counter += 1
                step_after_double_underscore.parent = prev_iter_output
                self._loop_body.compute()
                emit_type = self._add_to_emit_list(self._loop_body, emit_list,
                                                   emit_type)
                prev_iter_output = (
                    PrecomputedTraversal.as_precomputed(self._loop_body))
                self._loop_body.uncompute()
                until_mask = self._until_pred(prev_iter_output.last_vertices())

            # Fully reset the loop body so that this step can be recomputed
            # later.
            step_after_double_underscore.parent = __

            # Output all the paths that passed the emit predicate
            new_paths = pd.concat(emit_list)
            new_step_types = (
                self.parent.step_types
                + [prev_iter_output.step_types[-1]])
            # TODO: Raise NotImplementedError if different iterations emitted
            #  different types
            self._set_attrs(paths=new_paths,
                            step_types=new_step_types)


class CoalesceTraversal(UnaryTraversal):
    """
    A Gremlin `coalesce` step. Runs one or more traversals and returns the
    results of the first one that emits at least one result.
    """

    def __init__(self, parent: GraphTraversal,
                 subqueries: Sequence[GraphTraversal]):
        """
        :param parent: Input to the `coalesce` step
        :param subqueries: Sub-traversals to run in this step.
        """
        UnaryTraversal.__init__(self, parent)
        for s in subqueries:
            found_double_underscore, _ = find_double_underscore(s)
            if not found_double_underscore:
                raise NotImplementedError("coalesce without __ not implemented")
        self._subqueries = subqueries

    def compute_impl(self) -> None:
        # Add an artificial leading column so that we can tell duplicate paths
        # apart and emit the right number of duplicates on the output of this
        # step
        paths_with_leading_col = self.parent.paths.copy()
        paths_with_leading_col.insert(0, "artificial_leading_column",
                                      self.parent.paths.index)
        col_types_with_leading_col = ["a"] + self.parent.step_types
        aliases_with_leading_col = {
            k: v + 1 for k, v in self.parent.aliases.items()
        }

        # Loop until we run out of subqueries or input paths
        paths_list = []  # Type: List[pd.DataFrame]
        next_step_types = []  # Type: List[str]
        subquery_index = 0
        while (subquery_index < len(self._subqueries)
               and len(paths_with_leading_col.index) > 0):
            # TODO: The logic here is similar to WhereSubqueryTraversal. Break
            #  out the common code into a new common superclass.
            subquery = self._subqueries[subquery_index]
            _, step_after_double_underscore = find_double_underscore(subquery)
            double_underscore_replacement = (
                PrecomputedTraversal(vertices=self.parent.vertices,
                                     edges=self.parent.edges,
                                     paths=paths_with_leading_col,
                                     step_types=col_types_with_leading_col,
                                     aliases=aliases_with_leading_col))
            step_after_double_underscore.parent = double_underscore_replacement
            subquery.compute()
            subquery_paths = subquery.paths
            parent_path_len = len(self.parent.paths.columns)
            if len(subquery_paths.columns) == parent_path_len + 1:
                # Subquery didn't add any elements, just filtered parent paths.
                # Strip off the artificial leading column.
                paths_list.append(subquery_paths[1:])
                next_step_types.append("empty")  # Special flag for our own use
            else:
                # Subquery added at least one element to the paths.
                # Keep the parts of the path that came from the parent, plus the
                # last element that came from the subquery.
                columns_to_keep = subquery_paths.columns[
                    list(range(1, parent_path_len + 1)) + [-1]]
                paths = subquery_paths[columns_to_keep]
                # Clean up column names
                paths.columns = list(range(len(paths.columns)))
                paths_list.append(paths)
                next_step_types.append(subquery.step_types[-1])

            # Remove input paths that produced outputs on this iteration
            remaining_leading_col_values = subquery_paths[0].unique()
            paths_with_leading_col = paths_with_leading_col[
                ~paths_with_leading_col["artificial_leading_column"]
                    .isin(remaining_leading_col_values)]

            # Reset the subquery so that this step can be recomputed later.
            subquery.uncompute()
            step_after_double_underscore.parent = __
            subquery_index += 1

        for i in range(1, len(next_step_types)):
            if next_step_types[i] != next_step_types[0]:
                raise ValueError(f"Traversals 0 and {i} passed to coalesce "
                                 f"produce mismatched output element types "
                                 f"'{next_step_types[0]}' and "
                                 f"'{next_step_types[i]}'")
        new_step_types = self.parent.step_types.copy()
        if next_step_types[0] != "empty":
            new_step_types.append(next_step_types[0])
        new_paths = pd.concat(paths_list)
        self._set_attrs(paths=new_paths, step_types=new_step_types)


class UnionTraversal(UnaryTraversal):
    """
    A Gremlin `union` step. Runs one or more traversals and returns the
    *multiset* union of all their results.
    """

    def __init__(self, parent: GraphTraversal,
                 subqueries: Sequence[GraphTraversal]):
        """
        :param parent: Input to the `coalesce` step
        :param subqueries: Sub-traversals to run in this step.
        """
        UnaryTraversal.__init__(self, parent)
        for s in subqueries:
            found_double_underscore, _ = find_double_underscore(s)
            if not found_double_underscore:
                raise NotImplementedError("union without __ not implemented")
        self._subqueries = subqueries

    def compute_impl(self) -> None:
        # TODO: This code has a lot in common with the coalesce and where steps.
        #  Factor out the common parts.

        # Loop until we run out of subqueries or input paths
        paths_list = []  # Type: List[pd.DataFrame]
        next_step_types = []  # Type: List[str]
        for subquery in self._subqueries:
            _, step_after_double_underscore = find_double_underscore(subquery)
            double_underscore_replacement = (
                PrecomputedTraversal(vertices=self.parent.vertices,
                                     edges=self.parent.edges,
                                     paths=self.parent.paths,
                                     step_types=self.parent.step_types,
                                     aliases=self.parent.aliases))
            step_after_double_underscore.parent = double_underscore_replacement
            subquery.compute()
            subquery_paths = subquery.paths
            if len(subquery_paths.index) > 0:
                parent_path_len = len(self.parent.paths.columns)
                if len(subquery_paths.columns) == parent_path_len:
                    # Subquery didn't add any elements, just filtered parent paths.
                    paths_list.append(subquery_paths)
                    next_step_types.append("empty")  # Special flag for our own use
                else:
                    # Subquery added at least one element to the paths.
                    # Keep the parts of the path that came from the parent, plus the
                    # last element that came from the subquery.
                    columns_to_keep = subquery_paths.columns[
                        list(range(0, parent_path_len)) + [-1]]
                    paths = subquery_paths[columns_to_keep]
                    # Clean up column names
                    paths.columns = list(range(len(paths.columns)))
                    paths_list.append(paths)
                    next_step_types.append(subquery.step_types[-1])

            # Reset the subquery so that this step can be recomputed later.
            subquery.uncompute()
            step_after_double_underscore.parent = __

        for i in range(1, len(next_step_types)):
            if next_step_types[i] != next_step_types[0]:
                raise ValueError(f"Traversals 0 and {i} passed to union "
                                 f"produce mismatched output element types "
                                 f"'{next_step_types[0]}' and "
                                 f"'{next_step_types[i]}'")
        new_step_types = self.parent.step_types.copy()
        if next_step_types[0] != "empty":
            new_step_types.append(next_step_types[0])
        new_paths = pd.concat(paths_list)
        self._set_attrs(paths=new_paths, step_types=new_step_types)