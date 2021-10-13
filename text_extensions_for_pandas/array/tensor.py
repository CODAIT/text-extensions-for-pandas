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

from distutils.version import LooseVersion
import numbers
import os
from typing import *

import numpy as np
import pandas as pd
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
try:
    from pandas.core.dtypes.generic import ABCIndex
except ImportError:
    # ABCIndexClass changed to ABCIndex in Pandas 1.3
    # noinspection PyUnresolvedReferences
    from pandas.core.dtypes.generic import ABCIndexClass as ABCIndex
from pandas.core.indexers import check_array_indexer, validate_indices

""" Begin Patching of ExtensionArrayFormatter """
from pandas.io.formats.format import ExtensionArrayFormatter


def _format_strings_patched(self) -> List[str]:
    from pandas.core.construction import extract_array
    from pandas.io.formats.format import format_array

    if not isinstance(self.values, TensorArray):
        return self._format_strings_orig()

    values = extract_array(self.values, extract_numpy=True)
    array = np.asarray(values)

    if array.ndim == 1:
        return self._format_strings_orig()

    def format_array_wrap(array_, formatter_):
        fmt_values = format_array(
            array_,
            formatter_,
            float_format=self.float_format,
            na_rep=self.na_rep,
            digits=self.digits,
            space=self.space,
            justify=self.justify,
            decimal=self.decimal,
            leading_space=self.leading_space,
            quoting=self.quoting,
        )
        return fmt_values

    flat_formatter = self.formatter
    if flat_formatter is None:
        flat_formatter = values._formatter(boxed=True)

    # Flatten array, call function, reshape (use ravel_compat in v1.3.0)
    flat_array = array.ravel("K")
    fmt_flat_array = np.asarray(
        format_array_wrap(flat_array, flat_formatter))
    order = "F" if array.flags.f_contiguous else "C"
    fmt_array = fmt_flat_array.reshape(array.shape, order=order)

    # Format the array of nested strings, use default formatter
    return format_array_wrap(fmt_array, None)


def _format_strings_patched_v1_0_0(self) -> List[str]:
    from functools import partial
    from pandas.core.construction import extract_array
    from pandas.io.formats.format import format_array
    from pandas.io.formats.printing import pprint_thing

    if not isinstance(self.values, TensorArray):
        return self._format_strings_orig()

    values = extract_array(self.values, extract_numpy=True)
    array = np.asarray(values)

    if array.ndim == 1:
        return self._format_strings_orig()

    def format_array_wrap(array_, formatter_):
        fmt_values = format_array(
            array_,
            formatter_,
            float_format=self.float_format,
            na_rep=self.na_rep,
            digits=self.digits,
            space=self.space,
            justify=self.justify,
            decimal=self.decimal,
            leading_space=self.leading_space,
        )
        return fmt_values

    flat_formatter = self.formatter
    if flat_formatter is None:
        flat_formatter = values._formatter(boxed=True)

    # Flatten array, call function, reshape (use ravel_compat in v1.3.0)
    flat_array = array.ravel("K")
    fmt_flat_array = np.asarray(
        format_array_wrap(flat_array, flat_formatter))
    order = "F" if array.flags.f_contiguous else "C"
    fmt_array = fmt_flat_array.reshape(array.shape, order=order)

    # Slimmed down version of GenericArrayFormatter due to pandas-dev GH#33770
    def format_strings_slim(array_, leading_space):
        formatter = partial(
            pprint_thing,
            escape_chars=("\t", "\r", "\n"),
        )

        def _format(x):
            return str(formatter(x))

        fmt_values = []
        for v in array_:
            tpl = "{v}" if leading_space is False else " {v}"
            fmt_values.append(tpl.format(v=_format(v)))
        return fmt_values

    return format_strings_slim(fmt_array, self.leading_space)


_FORMATTER_ENABLED_KEY = "TEXT_EXTENSIONS_FOR_PANDAS_FORMATTER_ENABLED"

if os.getenv(_FORMATTER_ENABLED_KEY, "true").lower() == "true":
    ExtensionArrayFormatter._format_strings_orig = \
        ExtensionArrayFormatter._format_strings
    if LooseVersion("1.1.0") <= LooseVersion(pd.__version__) < LooseVersion("1.3.0"):
        ExtensionArrayFormatter._format_strings = _format_strings_patched
    else:
        ExtensionArrayFormatter._format_strings = _format_strings_patched_v1_0_0
    ExtensionArrayFormatter._patched_by_text_extensions_for_pandas = True
""" End Patching of ExtensionArrayFormatter """


@pd.api.extensions.register_extension_dtype
class TensorDtype(pd.api.extensions.ExtensionDtype):
    """
    Pandas data type for a column of tensors with the same shape.
    """
    base = None

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
        See docstring in :class:`ExtensionDType` class in ``pandas/core/dtypes/base.py``
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
        See docstring in :class:`ExtensionDType` class in ``pandas/core/dtypes/base.py``
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

            if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
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

        op_name = f"__{op.__name__}__"
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
    A Pandas :class:`ExtensionArray` that represents a column of :class:`numpy.ndarray`
    objects, or tensors, where the outer dimension is the count of tensors in the column.
    Each tensor must have the same shape.
    """

    def __init__(self, values: Union[np.ndarray, Sequence[Union[np.ndarray, TensorElement]],
                                     TensorElement, Any]):
        """
        :param values: A :class:`numpy.ndarray` or sequence of
            :class:`numpy.ndarray` objects of equal shape.
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
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
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
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        raise NotImplementedError

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence["TensorArray"]
    ) -> "TensorArray":
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        return TensorArray(np.concatenate([a._tensor for a in to_concat]))

    def isna(self) -> np.array:
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        if self._tensor.dtype.type is np.object_:
            # Avoid comparing with __eq__ because the elements of the tensor may do
            # something funny with that operation.
            result_list = [
                self._tensor[i] is None for i in range(len(self))
            ]
            return np.array(result_list, dtype=bool)
        elif self._tensor.dtype.type is np.str_:
            return np.all(self._tensor == "", axis=-1)
        else:
            return np.all(np.isnan(self._tensor), axis=-1)

    def copy(self) -> "TensorArray":
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
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
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
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
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        return TensorDtype()

    @property
    def inferred_type(self) -> str:
        """
        Return string describing type of TensorArray. Delegates to
        :func:`pandas.api.types.infer_dtype`. See docstring for more information.

        :return: string describing numpy type of this TensorArray
        """
        return pd.api.types.infer_dtype(self._tensor)

    @property
    def nbytes(self) -> int:
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        return self._tensor.nbytes

    def to_numpy(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default):
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
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
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        dtype = pd.api.types.pandas_dtype(dtype)

        if isinstance(dtype, TensorDtype):
            values = TensorArray(self._tensor.copy()) if copy else self
        elif not pd.api.types.is_object_dtype(dtype) and \
                pd.api.types.is_string_dtype(dtype):
            values = np.array([str(t) for t in self._tensor])
            if isinstance(dtype, pd.StringDtype):
                return dtype.construct_array_type()._from_sequence(values, copy=False)
            else:
                return values
        elif pd.api.types.is_object_dtype(dtype):
            # Interpret astype(object) as "cast to an array of numpy arrays"
            values = np.empty(len(self), dtype=object)
            for i in range(len(self)):
                values[i] = self._tensor[i]
        else:
            values = self._tensor.astype(dtype, copy=copy)
        return values

    def any(self, axis=None, out=None, keepdims=False):
        """
        Test whether any array element along a given axis evaluates to ``True``.

        See numpy.any() documentation for more information
        https://numpy.org/doc/stable/reference/generated/numpy.any.html#numpy.any

        :param axis: Axis or axes along which a logical OR reduction is performed.
        :param out: Alternate output array in which to place the result.
        :param keepdims: If this is set to True, the axes which are reduced are left in the
         result as dimensions with size one.

        :return: single boolean unless ``axis``is not ``None``; else :class:`TensorArray`
        """
        result = self._tensor.any(axis=axis, out=out, keepdims=keepdims)
        return result if axis is None else TensorArray(result)

    def all(self, axis=None, out=None, keepdims=False):
        """
        Test whether all array elements along a given axis evaluate to ``True``.

        :param axis: Axis or axes along which a logical AND reduction is performed.
        :param out: Alternate output array in which to place the result.
        :param keepdims: If this is set to True, the axes which are reduced are left in the
         result as dimensions with size one.

        :return: single boolean unless ``axis`` is not ``None``; else :class:`TensorArray`
        """
        result = self._tensor.all(axis=axis, out=out, keepdims=keepdims)
        return result if axis is None else TensorArray(result)

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, item) -> Union["TensorArray", "TensorElement"]:
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        # Return scalar if single value is selected, a TensorElement for single array
        # element, or TensorArray for slice
        if isinstance(item, int):
            value = self._tensor[item]
            if np.isscalar(value):
                return value
            else:
                return TensorElement(value)
        else:
            # BEGIN workaround for Pandas issue #42430
            if isinstance(item, tuple) and len(item) > 1 and item[0] == Ellipsis:
                if len(item) > 2:
                    # Hopefully this case is not possible, but can't be sure
                    raise ValueError(f"Workaround Pandas issue #42430 not implemented "
                                     f"for tuple length > 2")
                item = item[1]
            # END workaround for issue #42430
            if isinstance(item, TensorArray):
                item = np.asarray(item)
            item = check_array_indexer(self, item)
            return TensorArray(self._tensor[item])

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
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

    def __contains__(self, item) -> bool:
        if isinstance(item, TensorElement):
            npitem = np.asarray(item)
            if npitem.size == 1 and np.isnan(npitem).all():
                return self.isna().any()
        return super().__contains__(item)

    def __repr__(self):
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        return self._tensor.__repr__()

    def __str__(self):
        return self._tensor.__str__()

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
        for information about this method.
        """
        # TODO return self._tensor, np.nan
        raise NotImplementedError

    def _reduce(self, name, skipna=True, **kwargs):
        """
        See docstring in :class:`ExtensionArray` class in ``pandas/core/arrays/base.py``
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
        """
        Interface to return the backing tensor as a numpy array with optional dtype.
        If dtype is not None, then the tensor will be cast to that type, otherwise
        this is a no-op.
        """
        return np.asarray(self._tensor, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Interface to handle numpy ufuncs that will accept TensorArray as input, and wrap
        the output back as another TensorArray.
        """
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, (TensorArray, np.ndarray, numbers.Number)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x._tensor if isinstance(x, TensorArray) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x._tensor if isinstance(x, TensorArray) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __arrow_array__(self, type=None):
        from text_extensions_for_pandas.array.arrow_conversion import ArrowTensorArray
        return ArrowTensorArray.from_numpy(self._tensor)


# Add operators from the mixin to the class
TensorElement._add_arithmetic_ops()
TensorElement._add_comparison_ops()
TensorArray._add_arithmetic_ops()
TensorArray._add_comparison_ops()
