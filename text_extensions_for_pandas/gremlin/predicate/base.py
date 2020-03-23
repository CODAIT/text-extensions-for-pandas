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

# base.py
#
# Abstract base classes for predicates.
from abc import ABC
from typing import Iterator

import pandas as pd
import numpy as np
from typing import *

from text_extensions_for_pandas.gremlin.traversal.base import GraphTraversal


class VertexPredicate:
    """
    Base class for Boolean predicates applied to individual vertices of the
    graph.
    """

    def __init__(self, *children: "VertexPredicate"):
        self._children = children

    def __call__(self, vertices: pd.DataFrame) -> np.ndarray:
        """
        :param vertices: DataFrame of vertices on which to apply the predicate
        :return: A numpy Boolean mask containing `True` for each row that
        satisfies the predicate
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def bind_aliases(self, parent: GraphTraversal) -> None:
        """
        Bind any aliases to other parts of the current path to the appropriate
        vertices of the current path.
        :param parent: Tail node node in the current path. Must already be
        computed.
        """
        self.bind_aliases_self(parent)
        for c in self._children:
            c.bind_aliases(parent)

    def bind_aliases_self(self, parent: GraphTraversal) -> None:
        """
        Subclasses that reference other nodes of the current path should
        override this method.

        :param parent: Tail node node in the current path. Must already be
        computed.
        """
        pass

    def modulate(self, modulator: Iterator[str]) -> None:
        """
        Apply one or more modulators to this predicate.

        :param modulator: Infinite iterator backed by a circular buffer of
        string-valued modulators to apply round-robin style to the expression
        tree rooted at this node.
        """
        self.modulate_self(modulator)
        for c in self._children:
            c.modulate(modulator)

    def modulate_self(self, modulator: Iterator[str]) -> None:
        """
        Subclasses that consume an element of the modulators stream should
        override this method.

        :param modulator: Infinite iterator backed by a circular buffer of
        string-valued modulators to apply round-robin style to the expression
        tree rooted at this node.
        """
        pass


class ColumnPredicate(VertexPredicate, ABC):
    """
    Abstract base class for VertexPredicates that only read one column and may
    need that column to be bound late, as in a `has` step.
    """

    def __init__(self):
        VertexPredicate.__init__(self)
        self._target_col = None

    def modulate_self(self, modulator: Iterator[str]) -> None:
        self._target_col = next(modulator)

    @property
    def target_col(self) -> str:
        """
        :returns: Name of the column on which this predicate will be applied.
        """
        return self._target_col


class BinaryPredicate(VertexPredicate, ABC):
    """
    Abstract base class for Gremlin binary predicates.
    """

    def __init__(self, target_alias: str, *children: "BinaryPredicate"):
        """
        :param target_alias: Name of the second vertex to compare against.
        :param *children: Optional set of child predicates for propagating bindings
        """
        VertexPredicate.__init__(self, *children)
        self._target_alias = target_alias
        self._target_vertices = None  # Type: pd.DataFrame
        self._left_col = None  # Type: str
        self._right_col = None  # Type: str

    def bind_aliases_self(self, parent: GraphTraversal) -> None:
        self._target_vertices = parent.alias_to_vertices(self._target_alias)

    def modulate_self(self, modulator: Iterator[str]) -> None:
        self._left_col = next(modulator)
        self._right_col = next(modulator)

    @property
    def target_alias(self) -> str:
        """
        :return: Name of the alias that the
        """
        return self._target_alias

    @property
    def target_vertices(self) -> pd.DataFrame:
        """
        :return: The current set of vertices in the second argument of this
        predicate.
        """
        if self._target_vertices is None:
            raise ValueError(f"Attempted to get other_vertices property before "
                             f"calling bind_aliases_self on {self}")
        return self._target_vertices
