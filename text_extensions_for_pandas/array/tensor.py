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

    def __from_arrow__(self, extension_array):
        from text_extensions_for_pandas.array.arrow_conversion import arrow_to_tensor_array
        return arrow_to_tensor_array(extension_array)


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

    def __init__(self, values: Union[np.ndarray, Sequence[np.ndarray]],
                 make_contiguous: bool = True):
        """
        :param values: A `numpy.ndarray` or sequence of `numpy.ndarray`s of equal shape.
        :param make_contiguous: force values to be contiguous in memory (C order)
        """
        if isinstance(values, np.ndarray):
            self._tensor = values
        elif isinstance(values, Sequence):
            self._tensor = np.stack(values, axis=0) if len(values) > 0 else np.array([])
        elif isinstance(values, TensorArray):
            raise TypeError("Use the copy() method to create a copy of a TensorArray")
        else:
            raise TypeError(f"Expected a numpy.ndarray or sequence of numpy.ndarray, "
                            f"but received {values} "
                            f"of type '{type(values)}' instead.")
        
        if not self._tensor.flags.c_contiguous and make_contiguous:
            self._tensor = np.ascontiguousarray(self._tensor)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if copy and isinstance(scalars, np.ndarray):
            scalars = scalars.copy()
        elif isinstance(scalars, TensorArray):
            scalars = scalars._tensor.copy() if copy else scalars._tensor
        return TensorArray(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        raise NotImplementedError

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
        return np.all(np.isnan(self._tensor), axis=-1)

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

    @property
    def nbytes(self) -> int:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return self._tensor.nbytes

    def to_numpy(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        dtype = pd.api.types.pandas_dtype(dtype)
        if dtype is not None:
            if copy:
                values = np.array(self._tensor, dtype=dtype, copy=True)
            else:
                values = self._tensor.astype(dtype)
        elif copy:
            values = self._tensor.copy()
        else:
            values = self._tensor
        return values

    def astype(self, dtype, copy=True):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        dtype = pd.api.types.pandas_dtype(dtype)
        if isinstance(dtype, TensorType):
            values = TensorArray(self._tensor.copy() if copy else self._tensor)
        else:
            values = self._tensor.astype(dtype, copy=copy)
        return values

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, item) -> "TensorArray":
        """
        See docstring in `Extension   Array` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO pandas converts series with np.asarray, then applied a function e.g. map_infer(array, is_float) to format strings etc.
        # Return an ndarray for scalar item, or TensorArray for slice
        if isinstance(item, int):
            return self._tensor[item]
        else:
            return TensorArray(self._tensor[item])

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(key, (int, slice)):
            self._tensor[key] = value
        else:
            raise NotImplementedError(f"__setitem__ with key type '{type(key)}' "
                                      f"not implemented")

    def __repr__(self):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return self._tensor.__repr__()

    def __str__(self):
        return self._tensor.__str__()

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO return self._tensor, np.nan
        raise NotImplementedError

    def _reduce(self, name, skipna=True, **kwargs):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if name == "sum":
            return TensorArray(np.sum(self._tensor, axis=0))
        else:
            raise NotImplementedError(f"'{name}' aggregate not implemented.")

    def __arrow_array__(self, type=None):
        from text_extensions_for_pandas.array.arrow_conversion import ArrowTensorArray
        return ArrowTensorArray.from_numpy(self._tensor)


# Add operators from the mixin to the class
TensorArray._add_arithmetic_ops()
TensorArray._add_comparison_ops()
