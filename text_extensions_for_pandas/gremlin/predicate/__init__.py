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

# predicate.__init__.py
#
# Boolean predicates for Gremlin queries.

from typing import Any

import numpy as np
import pandas as pd

from text_extensions_for_pandas.gremlin.predicate.base import BinaryPredicate, \
    ColumnPredicate, VertexPredicate, BinaryPredicate


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
        ColumnPredicate.__init__(self)
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
        ColumnPredicate.__init__(self)
        self._args = args

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        return (~vertices[self.target_col].isin(self._args)).values


class LessThanPredicate(BinaryPredicate):
    """
    Implementation of the Gremlin `lt()` predicate.
    """

    def __init__(self, other: str):
        """
        :param other: Name of the second field to compare against
        """
        BinaryPredicate.__init__(self, other)

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        # The inputs are views on the vertices tables, so we need to reset the
        # Pandas indexes to prevent the lt operation below from matching pairs
        # of rows by (unused) index
        left_series = vertices[self._left_col].reset_index(drop=True)
        right_series = self.target_vertices[self._right_col].reset_index(
            drop=True)
        result_series = left_series.lt(right_series)
        return result_series.values


class OverlapsPredicate(BinaryPredicate):
    """
    *Extension to Gremlin.* A binary predicate that returns True if the
    two input spans overlap.
    """

    def __init__(self, other: str):
        """
        :param other: Name of the second field to compare against
        """
        BinaryPredicate.__init__(self, other)

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        # The inputs are views on the vertices tables, so we need to reset the
        # Pandas indexes to prevent the operation below from matching pairs
        # of rows by (unused) index
        left_series = vertices[self._left_col].reset_index(drop=True)
        right_series = self.target_vertices[self._right_col].reset_index(
            drop=True)
        result_array = left_series.values.overlaps(right_series.values)
        return result_array


class NotBinaryPredicate(BinaryPredicate):
    """
    Invert a binary predicate
    """
    def __init__(self, child: BinaryPredicate):
        BinaryPredicate.__init__(self, child.target_alias, child)
        self._child = child

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        return ~self._child(vertices)


########################################################
# Syntactic sugar to keep pep8 happy about class names


def within(*args):
    return Within(*args)


def without(*args):
    return Without(*args)


def lt(other):
    return LessThanPredicate(other)


def overlaps(other):
    return OverlapsPredicate(other)


def not_(other):
    if isinstance(other, BinaryPredicate):
        return NotBinaryPredicate(other)
    else:
        raise NotImplementedError("not_() over unary predicate not yet implemented")

# End syntactic sugar
########################################################
