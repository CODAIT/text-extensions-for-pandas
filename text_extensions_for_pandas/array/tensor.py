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
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices


@pd.api.extensions.register_extension_dtype
class TensorDtype(pd.api.extensions.ExtensionDtype):
    """
    Pandas data type for a column of tensors with the same shape.
    """

    @property
    def type(self):
        """The type for a single row of a TensorArray column."""
        return TensorElement

    @property
    def name(self) -> str:
        """A string representation of the dtype."""
        return "TensorDtype"

    @classmethod
    def construct_from_string(cls, string: str):
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
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
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        # NOTE: this overrides, but coerce_to_dtype, result_dtype might not be needed

        def _binop(self, other):
            lvalues = self._tensor

            if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndexClass)):
                # Rely on pandas to unbox and dispatch to us.
                return NotImplemented

            # divmod returns a tuple
            if op_name in ["__divmod__", "__rdivmod__"]:
                # TODO: return tuple
                # div, mod = result
                raise NotImplementedError

            if isinstance(other, (TensorArray, TensorElement)):
                rvalues = other._tensor
            else:
                rvalues = other

            result = op(lvalues, rvalues)

            # Force a TensorArray if rvalue is not a scalar
            if isinstance(self, TensorElement) and \
                    (not isinstance(other, TensorElement) or not np.isscalar(other)):
                result_wrapped = TensorArray(result)
            else:
                result_wrapped = cls(result)

            return result_wrapped

        op_name = ops._get_op_name(op, True)
        return set_function_name(_binop, op_name, cls)


class TensorElement(TensorOpsMixin):
    """
    Class representing a single element in a TensorArray, or row in a Pandas column of dtype
    TensorDtype. This is a light wrapper over a numpy.ndarray
    """
    def __init__(self, values: np.ndarray):
        """
        Construct a TensorElement from an numpy.ndarray.
        :param values: tensor values for this instance.
        """
        self._tensor = values

    def __repr__(self):
        return self._tensor.__repr__()

    def __str__(self):
        return self._tensor.__str__()

    def to_numpy(self):
        """
        Return the values of this element as a numpy.ndarray
        :return: numpy.ndarray
        """
        return np.asarray(self._tensor)

    def __array__(self):
        return np.asarray(self._tensor)


class TensorArray(pd.api.extensions.ExtensionArray, TensorOpsMixin):
    """
    A Pandas `ExtensionArray` that represents a column of `numpy.ndarray`s,
    or tensors, where the outer dimension is the count of tensors in the column.
    Each tensor must have the same shape.
    """

    def __init__(self, values: Union[np.ndarray, Sequence[Union[np.ndarray, TensorElement]],
                                     TensorElement, Any]):
        """
        :param values: A `numpy.ndarray` or sequence of `numpy.ndarray`s of equal shape.
        """
        if isinstance(values, np.ndarray):
            if values.dtype.type is np.object_ and len(values) > 0 and \
                    isinstance(values[0], TensorElement):
                self._tensor = np.array([np.asarray(v) for v in values])
            else:
                self._tensor = values
        elif isinstance(values, Sequence):
            if len(values) == 0:
                self._tensor = np.array([])
            else:
                self._tensor = np.stack([np.asarray(v) for v in values], axis=0)
        elif isinstance(values, TensorElement):
            self._tensor = np.array([np.asarray(values)])
        elif np.isscalar(values):
            # `values` is a single element: pd.Series(np.nan, index=[1, 2, 3], dtype=TensorDtype())
            self._tensor = np.array([values])
        elif isinstance(values, TensorArray):
            raise TypeError("Use the copy() method to create a copy of a TensorArray")
        else:
            raise TypeError(f"Expected a numpy.ndarray or sequence of numpy.ndarray, "
                            f"but received {values} "
                            f"of type '{type(values)}' instead.")

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
        if self._tensor.dtype.type is np.object_:
            return self._tensor == None
        elif self._tensor.dtype.type is np.str_:
            return np.all(self._tensor == "", axis=-1)
        else:
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
        if allow_fill:
            # From API docs: "[If allow_fill == True, then] negative values in
            # `indices` indicate missing values and are set to `fill_value`
            indices = np.asarray(indices, dtype=np.intp)
            validate_indices(indices, len(self._tensor))

            # Check if there are missing indices to fill, if not can use numpy take below
            has_missing = np.any(indices < 0)
            if has_missing:
                if fill_value is None:
                    fill_value = np.nan
                # Create an array populated with fill value
                values = np.full((len(indices),) + self._tensor.shape[1:], fill_value)

                # Iterate over each index and set non-missing elements
                for i, idx in enumerate(indices):
                    if idx >= 0:
                        values[i] = self._tensor[idx]
                return TensorArray(values)

        # Delegate take to numpy array
        values = self._tensor.take(indices, axis=0)

        return TensorArray(values)

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return TensorDtype()

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
        if dtype is not None:
            dtype = pd.api.types.pandas_dtype(dtype)
            if copy:
                values = np.array(self._tensor, dtype=dtype, copy=True)
            else:
                values = self._tensor.astype(dtype)
        elif copy:
            values = self._tensor.copy()
        else:
            values = self._tensor
        return values

    @property
    def numpy_dtype(self):
        """
        Get the dtype of the tensor.
        :return: The numpy dtype of the backing ndarray
        """
        return self._tensor.dtype

    @property
    def numpy_ndim(self):
        """
        Get the number of tensor dimensions.
        :return: integer for the number of dimensions
        """
        return self._tensor.ndim

    @property
    def numpy_shape(self):
        """
        Get the shape of the tensor.
        :return: A tuple of integers for the numpy shape of the backing ndarray
        """
        return self._tensor.shape

    def astype(self, dtype, copy=True):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        dtype = pd.api.types.pandas_dtype(dtype)

        if isinstance(dtype, TensorDtype):
            values = TensorArray(self._tensor.copy() if copy else self._tensor)
        elif not pd.api.types.is_object_dtype(dtype) and \
                pd.api.types.is_string_dtype(dtype):
            values = np.array([str(t) for t in self._tensor])
            if isinstance(dtype, pd.StringDtype):
                return dtype.construct_array_type()._from_sequence(values, copy=False)
            else:
                return values
        else:
            values = self._tensor.astype(dtype, copy=copy)
        return values

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, item) -> Union["TensorArray", "TensorElement"]:
        """
        See docstring in `Extension   Array` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # Return scalar if single value is selected, a TensorElement for single array element,
        # or TensorArray for slice
        if isinstance(item, int):
            value = self._tensor[item]
            if np.isscalar(value):
                return value
            else:
                return TensorElement(value)
        else:
            item = check_array_indexer(self, item)
            return TensorArray(self._tensor[item])

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        key = check_array_indexer(self, key)
        if isinstance(value, TensorElement) or np.isscalar(value):
            value = np.asarray(value)
        if isinstance(value, list):
            value = [np.asarray(v) if isinstance(v, TensorElement) else v for v in value]
        if isinstance(value, ABCSeries) and isinstance(value.dtype, TensorDtype):
            value = value.values
        if value is None or isinstance(value, Sequence) and len(value) == 0:
            nan_fill = np.full_like(self._tensor[key], np.nan)
            self._tensor[key] = nan_fill
        elif isinstance(key, (int, slice, np.ndarray)):
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
            return TensorElement(np.sum(self._tensor, axis=0))
        elif name == "all":
            return TensorElement(np.all(self._tensor, axis=0))
        elif name == "any":
            return TensorElement(np.any(self._tensor, axis=0))
        else:
            raise NotImplementedError(f"'{name}' aggregate not implemented.")

    def __array__(self, dtype=None):
        return np.asarray(self._tensor, dtype=dtype)

    def __arrow_array__(self, type=None):
        from text_extensions_for_pandas.array.arrow_conversion import ArrowTensorArray
        return ArrowTensorArray.from_numpy(self._tensor)


# Add operators from the mixin to the class
TensorElement._add_arithmetic_ops()
TensorElement._add_comparison_ops()
TensorArray._add_arithmetic_ops()
TensorArray._add_comparison_ops()
