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

from abc import ABC
from typing import Iterator, Any

import numpy as np
import pandas as pd

from text_extensions_for_pandas.gremlin.traversal.base import GraphTraversalBase
from text_extensions_for_pandas.gremlin.predicate.base import ColumnPredicate, VertexPredicate


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


class BinaryPredicate(VertexPredicate, ABC):
    """
    Abstract base class for Gremlin binary predicates.
    """

    def __init__(self, other: str):
        """
        :param other: Name of the second vertex to compare against.
        """
        VertexPredicate.__init__(self)
        self._other_alias = other
        self._other_vertices = None  # Type: pd.DataFrame
        self._left_col = None  # Type: str
        self._right_col = None  # Type: str

    def bind_aliases_self(self, parent: GraphTraversalBase) -> None:
        self._other_vertices = parent.alias_to_vertices(self._other_alias)

    def modulate_self(self, modulator: Iterator[str]) -> None:
        self._left_col = next(modulator)
        self._right_col = next(modulator)

    @property
    def other_vertices(self) -> pd.DataFrame:
        """
        :return: The current set of vertices in the second argument of this
        predicate.
        """
        if self._other_vertices is None:
            raise ValueError(f"Attempted to get other_vertices property before "
                             f"calling bind_aliases_self on {self}")
        return self._other_vertices


class LessThanPredicate(BinaryPredicate):
    """
    Implementation of the Gremlin `lt()` predicate.
    """

    def __init__(self, other: str):
        """
        :param other: Name of the second vertex to compare against
        """
        BinaryPredicate.__init__(self, other)

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        # The inputs are views on the vertices tables, so we need to reset the
        # Pandas indexes to prevent the lt operation below from matching pairs
        # of rows by (unused) index
        left_series = vertices[self._left_col].reset_index(drop=True)
        right_series = self.other_vertices[self._right_col].reset_index(
            drop=True)
        result_series = left_series.lt(right_series)
        return result_series.values

########################################################
# Syntactic sugar to keep pep8 happy about class names


def within(*args):
    return Within(*args)


def without(*args):
    return Without(*args)


def lt(other):
    return LessThanPredicate(other)

# End syntactic sugar
########################################################
