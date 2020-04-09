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
from pandas.compat import set_function_name
from pandas.core import ops

# Internal imports
import text_extensions_for_pandas.util as util


@pd.api.extensions.register_extension_dtype
class TensorType(pd.api.extensions.ExtensionDtype):
    """
    Pandas data type for a column of tensors with the same shape.
    """

    @property
    def type(self):
        """The type for a single row of a TensorArray column."""
        return np.ndarray

    @property
    def name(self) -> str:
        """A string representation of the dtype."""
        return "TensorType"

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


class TensorOpsMixin(pd.api.extensions.ExtensionScalarOpsMixin):
    """
    Mixin to provide operators on underlying ndarray.
    TODO: would be better to derive from ExtensionOpsMixin, but not available
    """

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        # NOTE: this overrides, but coerce_to_dtype might not be needed

        def _binop(self, other):
            lvalues = self._tensor
            rvalues = other._tensor if isinstance(other, TensorArray) else other
            res = op(lvalues, rvalues)
            return TensorArray(res)

        op_name = ops._get_op_name(op, True)
        return set_function_name(_binop, op_name, cls)


class TensorArray(pd.api.extensions.ExtensionArray, TensorOpsMixin):
    """
    A Pandas `ExtensionArray` that represents a column of `numpy.ndarray`s,
    or tensors, where the outer dimension is the count of tensors in the column.
    Each tensor must have the same shape.
    """

    def __init__(self, values: Union[np.ndarray, Iterable[np.ndarray]],
                 make_contiguous: bool = True):
        """
        :param values: A `numpy.ndarray` or sequence of `numpy.ndarray`s of equal shape.
        :param make_contiguous: force values to be contiguous in memory
        """
        if isinstance(values, Iterable):
            self._tensor = np.stack(values, axis=0)
        elif isinstance(values, np.ndarray):
            self._tensor = values
        else:
            raise TypeError(f"Expected a numpy.ndarray or list of numpy.ndarray, "
                            f"but received {values} "
                            f"of type '{type(values)}' instead.")
        
        if not self._tensor.flags.c_contiguous and make_contiguous:
            self._tensor = np.ascontiguousarray(self._tensor)

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence["TensorArray"]
    ) -> "TensorArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return TensorArray(np.concatenate([a._tensor for a in to_concat]))

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
            self._tensor.copy(),
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
        values = self._tensor.take(indices, axis=0)
        if allow_fill:
            # From API docs: "[If allow_fill == True, then] negative values in
            # `indices` indicate missing values. These values are set to
            # `fill_value`.
            for i in range(len(indices)):
                if indices[i] < 0:
                    # Note that Numpy will broadcast the fill value to the shape
                    # of each row.
                    values[i] = fill_value
        return TensorArray(values)

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return TensorType()

    def to_numpy(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO options
        return self._tensor

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, item) -> "TensorArray":
        """
        See docstring in `Extension   Array` class in `pandas/core/arrays/base.py`
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

    def __str__(self):
        return self._tensor.__str__()

    def _reduce(self, name, skipna=True, **kwargs):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        raise NotImplementedError("not implemented")


# Add operators from the mixin to the class
TensorArray._add_arithmetic_ops()
TensorArray._add_comparison_ops()