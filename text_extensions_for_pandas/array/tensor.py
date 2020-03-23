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

#
# tensor.py
#
# Part of text_extensions_for_pandas
#
# Pandas extensions to support columns of N-dimensional tensors of equal shape.
#

from typing import *

import numpy as np
import pandas as pd
from memoized_property import memoized_property

# Internal imports
import text_extensions_for_pandas.util as util


@pd.api.extensions.register_extension_dtype
class TensorType(pd.api.extensions.ExtensionDtype):
    """
    Panda datatype for a span that represents a range of characters within a
    target string.
    """

    @property
    def type(self):
        # The type for a single row of a column of type CharSpan
        return np.ndarray

    @property
    def name(self) -> str:
        """A string representation of the dtype."""
        return "Tensor"

    @classmethod
    def construct_from_string(cls, string: str):
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        # Upstream code uses exceptions as part of its normal control flow and
        # will pass this method bogus class names.
        if string == cls.__name__:
            return cls()
        else:
            raise TypeError(
                f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        return TensorArray


class TensorArray(pd.api.extensions.ExtensionArray):
    """
    A Pandas `ExtensionArray` that represents a column of character-based spans
    over a single target text.

    Spans are represented as `[begin, end)` intervals, where `begin` and `end`
    are character offsets into the target text.
    """

    def __init__(self, values: Union[np.ndarray, List[np.ndarray]],
                 make_contiguous: bool=True):
        """
        :param text: Target text from which the spans of this array are drawn
        :param begins: Begin offsets of spans (closed)
        :param ends: End offsets (open)
        """
        if isinstance(values, list):
            self._tensor = np.stack(values, axis=0)
        else:
            self._tensor = values
        
        if not self._tensor.flags.c_contiguous:
            if make_contiguous:
                self._tensor = np.ascontiguousarray(self._tensor)
            else:
                raise ValueError("Input ndarray is not contiguous")

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence["TensorArray"]
    ) -> "TensorArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return TensorArray([a._tensor for a in to_concat])

    def isna(self) -> np.array:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO any or all values in row nan?
        return np.any(np.isnan(self._tensor), axis=1)

    def copy(self) -> "TensorArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        ret = TensorArray(
            self._tensor,
        )
        # TODO: Copy cached properties too
        return ret

    def take(
        self, indices: Sequence[int], allow_fill: bool = False,
        fill_value: Any = None
    ) -> "TensorArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO if allow_fill:
        values = self._tensor.take(indices, axis=0)
        return TensorArray(values)

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return TensorType()

    def __len__(self) -> int:
        return len(self._tensor)

    def __eq__(self, other):
        return self._tensor == other._tensor

    def __lt__(self, other):
        """
        Pandas-style array/series comparison function.

        :param other: Second operand of a Pandas "<" comparison with the series
        that wraps this TokenSpanArray.

        :return: Returns a boolean mask indicating which rows are less than
         `other`. span1 < span2 if span1.end <= span2.begin.
        """
        return self._tensor < other._tensor

    def __gt__(self, other):
        return self._tensor > other._tensor

    def __le__(self, other):
        return self._tensor <= other._tensor

    def __ge__(self, other):
        return self._tensor >= other._tensor

    def __getitem__(self, item) -> "TensorArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO pandas converts series with np.asarray, then applied a function e.g. map_infer(array, is_float) to format strings etc.
        # If single element return as np.ndarray, if slice return TensorArray of slice
        if isinstance(item, int):
            return self._tensor[item]
        else:
            return TensorArray(self._tensor[item])

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        raise NotImplementedError("not implemented")

    def __repr__(self):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return self._tensor.__repr__()

    def _reduce(self, name, skipna=True, **kwargs):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        raise NotImplementedError("not implemented")

    '''
    @property
    def values(self) -> np.ndarray:
        return self._tensor
    '''

    '''
    def as_frame(self) -> pd.DataFrame:
        """
        Returns a dataframe representation of this column based on Python
        atomic types.
        """
        return pd.DataFrame({
            "begin": self.begin,
            "end": self.end,
            "covered_text": self.covered_text
        })
    '''

    def _repr_html_(self) -> str:
        """
        HTML pretty-printing of a series of spans for Jupyter notebooks.
        """
        return util.pretty_print_html(self)

    def to_numpy(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO options
        return self._tensor
