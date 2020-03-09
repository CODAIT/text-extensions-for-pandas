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

# underscore.py
#
# Code associated with the Gremlin double-underscore (__) operator
from typing import Tuple

from text_extensions_for_pandas.gremlin.traversal.base import GraphTraversal, \
    UnaryTraversal
from text_extensions_for_pandas.gremlin.traversal.constant import \
    PrecomputedTraversal, BootstrapTraversal


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


__ = DoubleUnderscore()  # Alias to allow "pt.__" in Gremlin expressions


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