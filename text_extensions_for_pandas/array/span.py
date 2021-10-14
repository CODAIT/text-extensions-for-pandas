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
# span.py
#
# Part of text_extensions_for_pandas
#
# Pandas extensions to support columns of spans with character offsets.
#

import collections.abc
import textwrap
from typing import *

import numpy as np
import pandas as pd
from memoized_property import memoized_property
# noinspection PyProtectedMember
from pandas.api.types import is_bool_dtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
try:
    from pandas.core.dtypes.generic import ABCIndex
except ImportError:
    # ABCIndexClass changed to ABCIndex in Pandas 1.3
    # noinspection PyUnresolvedReferences
    from pandas.core.dtypes.generic import ABCIndexClass as ABCIndex
from pandas.core.indexers import check_array_indexer

# Internal imports
import text_extensions_for_pandas.jupyter as jupyter
from text_extensions_for_pandas.array.string_table import StringTable
from text_extensions_for_pandas.util import to_int_array


def _check_same_text(obj1, obj2):
    if isinstance(obj1, Span) and isinstance(obj2, Span):
        if obj1.target_text != obj1.target_text:
            raise ValueError(
                f"Spans are over different target text "
                f"(got {obj1.target_text} and {obj2.target_text})"
            )
        return
    if not (isinstance(obj1, SpanArray) or isinstance(obj2, SpanArray)):
        raise TypeError(f"Expected some combination of Span and SpanArray, "
                        f"but received {type(obj1)} and {type(obj2)}")

    same_text_mask = (
        obj1.same_target_text(obj2) if isinstance(obj1, SpanArray)
        else obj2.same_target_text(obj1))
    if not np.all(same_text_mask):
        raise ValueError(
            f"SpanArrays are over different target text "
            f"(got {obj1.same_target_text} and {obj2.same_target_text})\n"
            f"Comparison result: {same_text_mask}"
        )


class SpanOpMixin:
    """
    Mixin class to define common operations between Span and SpanArray.
    """

    def __add__(self, other) -> Union["Span", "SpanArray"]:
        """
        Add a pair of spans and/or span arrays.

        span1 + span2 == minimal span that covers both spans
        :param other: Span or SpanArray
        :return: minimal span (or array of spans) that covers both inputs.
        """
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented

        if isinstance(self, Span) and isinstance(other, Span):
            # Span + *Span = Span
            _check_same_text(self, other)
            return Span(self.target_text, min(self.begin, other.begin),
                        max(self.end, other.end))
        elif isinstance(self, (Span, SpanArray)) and isinstance(other, (Span, SpanArray)):
            # SpanArray + *Span* = SpanArray
            _check_same_text(self, other)
            return SpanArray(self.target_text,
                                    np.minimum(self.begin, other.begin),
                                    np.maximum(self.end, other.end))
        else:
            raise TypeError(f"Unexpected combination of span types for add operation: "
                            f"{type(self)} and {type(other)}")


class Span(SpanOpMixin):
    """
    Python object representation of a single span with character offsets; that
    is, a single row of a `SpanArray`.

    An offset of `Span.NULL_OFFSET_VALUE` (currently -1) indicates
    "not a span" in the sense that NaN is "not a number".

    Most of the methods and properties of this class are single-span versions of the
     eponymous methods in :class:`SpanArray`. See that class for API documentation.
    """

    # Begin/end value that indicates "not a span" in the sense that NaN is
    # "not a number".
    NULL_OFFSET_VALUE = -1  # Type: int

    def __init__(self, text: str, begin: int, end: int):
        """
        Args:
            text: target document text on which the span is defined
            begin: Begin offset (inclusive) within `text`
            end: End offset (exclusive, one past the last char) within `text`
        """
        if text is not None and not isinstance(text, str):
            raise TypeError(f"Text must be a string. Got {text} of type {type(text)}.")
        if Span.NULL_OFFSET_VALUE == begin:
            if Span.NULL_OFFSET_VALUE != end:
                raise ValueError("Begin offset with special 'null' value {} "
                                 "must be paired with an end offset of {}",
                                 Span.NULL_OFFSET_VALUE,
                                 Span.NULL_OFFSET_VALUE)
        elif begin < 0:
            raise ValueError("begin must be >= 0")
        elif end < 0:
            raise ValueError("end must be >= 0")
        elif end > len(text):
            raise ValueError(f"end must be less than length of target string "
                             f"({end} > {len(text)}")
        self._text = text
        self._begin = begin
        self._end = end

    def __repr__(self) -> str:
        if self.begin == Span.NULL_OFFSET_VALUE:
            return "NA"
        elif self.target_text is None:
            return f"[{self.begin}, {self.end}): None"
        else:
            return f"[{self.begin}, {self.end}): " \
                   f"'{textwrap.shorten(self.covered_text, 80)}'"

    def __eq__(self, other):
        if isinstance(other, Span):
            return (
                # All NAs considered equal
                (self.begin == Span.NULL_OFFSET_VALUE
                 and other.begin == Span.NULL_OFFSET_VALUE)
                or
                (self.begin == other.begin
                    and self.end == other.end
                    and self.target_text == other.target_text))
        elif isinstance(other, SpanArray):
            return other == self
        else:
            # Different type ==> not equal
            return False

    def __hash__(self):
        result = hash((self.target_text, self.begin, self.end))
        return result

    def __lt__(self, other):
        """
        span1 < span2 if span1.end <= span2.begin and both spans are over the same
        target text
        """
        if not isinstance(other, (Span, SpanArray)):
            raise ValueError(f"Less-than relationship not defined for {self} and {other} "
                             f"of types {type(self)} and {type(other)}.")
        elif isinstance(other, Span) and self.target_text != other.target_text:
            raise ValueError(f"Less-than relationship undefined for different target "
                             f"texts.")
        elif isinstance(other, SpanArray) and np.any(self.target_text
                                                     != other.target_text):
            raise ValueError(f"Less-than relationship undefined for different target "
                             f"texts. Indexes that differ are "
                             f"{np.argmin(self.target_text != other.target_text)}.")
        else:
            return self.end <= other.begin

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return other <= self

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    @property
    def target_text(self):
        return self._text

    @memoized_property
    def covered_text(self):
        """
        Returns the substring of `self.target_text` that this `Span`
        represents.
        """
        if Span.NULL_OFFSET_VALUE == self._begin:
            return None
        else:
            return self.target_text[self.begin:self.end]

    def overlaps(self, other: "Span"):
        """
        :param other: Another Span or TokenSpan
        :return: True if the two spans overlap. Also True if a zero-length
            span is contained within the other.
        """
        if self.target_text != other.target_text:
            return False
        elif self.begin == other.begin and self.end == other.end:
            # Ensure that pairs of identical zero-length spans overlap.
            return True
        elif other.begin >= self.end:
            return False  # other completely to the right of self
        elif other.end <= self.begin:
            return False  # other completely to the left of self
        else:  # other.begin < self.end and other.end >= self.begin
            return True

    def contains(self, other: "Span"):
        """
        :param other: Another Span or TokenSpan
        :return: True if `other` is entirely within the bounds of this span. Also
            True if a zero-length span is contained within the other.
        """
        if self.target_text != other.target_text:
            return False
        return other.begin >= self.begin and other.end <= self.end

    def context(self, num_chars: int = 40) -> str:
        """
        Show the location of this span in the context of the target string.

        :param num_chars: How many characters on either side to display
        :return: A string in the form:
         ```<text before>[<text inside>]<text after>```
         describing the text within and around the span.
        """
        before_text = self.target_text[self.begin - num_chars:self.begin]
        after_text = self.target_text[self.end:self.end + num_chars]
        if self.begin > num_chars:
            before_text = "..." + before_text
        if self.end + num_chars < len(self.target_text):
            after_text = after_text + "..."
        return f"{before_text}[{self.covered_text}]{after_text}"


@pd.api.extensions.register_extension_dtype
class SpanDtype(pd.api.extensions.ExtensionDtype):
    """
    Panda datatype for a span that represents a range of characters within a
    target string.
    """
    @property
    def type(self):
        # The type for a single row of a column of type Span
        return Span

    @property
    def name(self) -> str:
        """A string representation of the dtype."""
        return "SpanDtype"

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
        return SpanArray

    @property
    def na_value(self) -> object:
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        return _NULL_SPAN_SINGLETON

    def __from_arrow__(self, extension_array):
        """
        Convert the given extension array of type ArrowSpanType to a
        SpanArray.
        """
        from text_extensions_for_pandas.array.arrow_conversion import arrow_to_span

        return arrow_to_span(extension_array)


_NULL_SPAN_SINGLETON = Span("", Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE)

_EMPTY_INT_ARRAY = np.zeros(0, dtype=int)


class SpanArray(pd.api.extensions.ExtensionArray, SpanOpMixin):
    """
    A Pandas `ExtensionArray` that represents a column of character-based spans
    over a single target text.

    Spans are represented as `[begin, end)` intervals, where `begin` and `end`
    are character offsets into the target text.
    """

    def __init__(self,
                 text: Union[str, Sequence[str], np.ndarray,
                             Tuple[StringTable, np.ndarray]],
                 begins: Union[pd.Series, np.ndarray, Sequence[int]],
                 ends: Union[pd.Series, np.ndarray, Sequence[int]]
                 ):
        """
        Factory method for creating instances of this class.

        :param text: Target text from which the spans of this array are drawn,
         or a sequence of texts if different spans can have different targets
        :param begins: Begin offsets of spans (closed)
        :param ends: End offsets (open)
        :return: A new `SpanArray` object
        """
        if not isinstance(begins, (pd.Series, np.ndarray, list)):
            raise TypeError(f"begins is of unsupported type {type(begins)}. "
                            f"Supported types are Series, ndarray and List[int].")
        if not isinstance(ends, (pd.Series, np.ndarray, list)):
            raise TypeError(f"ends is of unsupported type {type(ends)}. "
                            f"Supported types are Series, ndarray and List[int].")
        if len(begins) != len(ends):
            raise ValueError(f"Received {len(begins)} begin offsets and {len(ends)} "
                             f"offsets. Lengths should be equal.")
        begins = to_int_array(begins)
        ends = to_int_array(ends)

        if isinstance(text, str):
            # With a single string, every row gets string ID 0
            string_table = StringTable.create_single(text)  # type: StringTable
            text_ids = np.zeros_like(begins)  # type: np.ndarray
        elif isinstance(text, tuple):
            # INTERNAL USE ONLY: String table specified directly.
            # Note that this branch MUST come before the branch that checks for
            # sequences of strings, because tuples are sequences.
            string_table, text_ids = text
        elif isinstance(text, (collections.abc.Sequence, np.ndarray)):
            if len(text) != len(begins):  # Checked len(begins) == len(ends) earlier
                raise ValueError(f"Received {len(text)} target text values and "
                                 f"{len(begins)} begin offsets. Lengths should be equal.")
            string_table, text_ids = StringTable.merge_things(text)

        else:
            raise TypeError(f"Text argument is of unsupported type {type(text)}")

        # Begin and end offsets in characters
        self._begins = begins  # type: np.ndarray
        self._ends = ends  # type: np.ndarray

        self._string_table = string_table  # type: Union[StringTable, None]
        self._text_ids = text_ids

        # Cached list of other SpanArrays that are exactly the same as this
        # one. Each element is the result of calling id()
        self._equivalent_arrays = []  # type: List[int]

        # Version numbers of elements in self._equivalent_arrays, to ensure that
        # a change hasn't made the arrays no longer equal
        self._equiv_array_versions = []  # type: List[int]

        # Monotonically increasing version number for tracking changes and
        # invalidating caches
        self._version = 0

        # Flag that tells whether to display details of offsets in Jupyter notebooks
        self._repr_html_show_offsets = True  # type: bool

    ##########################################
    # Overrides of superclass methods go here.

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return SpanDtype()

    def astype(self, dtype, copy=True):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        dtype = pd.api.types.pandas_dtype(dtype)

        if isinstance(dtype, SpanDtype):
            data = self.copy() if copy else self
        elif isinstance(dtype, pd.StringDtype):
            # noinspection PyProtectedMember
            return dtype.construct_array_type()._from_sequence(self, copy=False)
        else:
            data = self.to_numpy(dtype=dtype, copy=copy, na_value=_NULL_SPAN_SINGLETON)
        return data

    @property
    def nbytes(self) -> int:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return (
            self._begins.nbytes + self._ends.nbytes + self._text_ids.nbytes
            + self._string_table.nbytes()
        )

    def __len__(self) -> int:
        return len(self._begins)

    def __getitem__(self, item) -> Union[Span, "SpanArray"]:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(item, int):
            return Span(self.target_text[item], int(self._begins[item]),
                        int(self._ends[item]))
        else:
            # item not an int --> assume it's a numpy-compatible index
            item = check_array_indexer(self, item)
            return SpanArray(
                (self._string_table, self._text_ids[item]),
                self._begins[item], self._ends[item])

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # Subroutine of the if-else sequence below
        def _is_sequence_of_spans(seq: Any):
            if isinstance(seq, SpanArray):
                return True
            if not isinstance(seq, (collections.abc.Sequence, np.ndarray)):
                return False
            else:
                # For other sequences, check for everything being Span or None
                return all(elem is None or isinstance(elem, Span) for elem in seq)

        key = check_array_indexer(self, key)
        if isinstance(value, ABCSeries) and isinstance(value.dtype, SpanDtype):
            value = value.values
        if value is None or (isinstance(value, collections.abc.Sequence)
                             and len(value) == 0):
            self._begins[key] = Span.NULL_OFFSET_VALUE
            self._ends[key] = Span.NULL_OFFSET_VALUE
            self._text_ids[key] = StringTable.NONE_ID
        elif isinstance(value, Span):
            self._begins[key] = value.begin
            self._ends[key] = value.end
            self._text_ids[key] = self._string_table.maybe_add_thing(value.target_text)
        elif ((isinstance(key, slice) or
               (isinstance(key, np.ndarray) and is_bool_dtype(key.dtype)))
              and isinstance(value, SpanArray)):
            self._begins[key] = value.begin
            self._ends[key] = value.end
            self._text_ids[key] = self._string_table.maybe_add_things(value.target_text)
        elif (isinstance(key, np.ndarray) and len(value) > 0 and len(value) == len(key)
              and _is_sequence_of_spans(value)):
            for k, v in zip(key, value):
                self._begins[k] = v.begin
                self._ends[k] = v.end
                self._text_ids[k] = self._string_table.maybe_add_thing(v.target_text)
        else:
            raise ValueError(
                f"Attempted to set element {key} (type {type(key)}) of a SpanArray with "
                f"an object of type {type(value)}")
        # We just changed the contents of this array, so invalidate any cached
        # results computed from those contents.
        self.increment_version()

    def __eq__(self, other):
        """
        Pandas/Numpy-style array/series comparison function.

        :param other: Second operand of a Pandas "==" comparison with the series
        that wraps this TokenSpanArray.

        :return: Returns a boolean mask indicating which rows match `other`.
        """
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        if isinstance(other, Span):
            mask = np.full(len(self), True, dtype=bool)
            mask[self.target_text != other.target_text] = False
            mask[self.begin != other.begin] = False
            mask[self.end != other.end] = False
            return mask
        elif isinstance(other, SpanArray):
            if len(self) != len(other):
                raise ValueError("Can't compare arrays of differing lengths "
                                 "{} and {}".format(len(self), len(other)))
            return np.logical_and(
                self.target_text == other.target_text,
                np.logical_and(
                    self.begin == other.begin,
                    self.end == other.end
                )
            )
        else:
            # TODO: Return False here once we're sure that this
            #  function is catching all the comparisons that really matter.
            raise ValueError("Don't know how to compare objects of type "
                             "'{}' and '{}'".format(type(self), type(other)))

    def __ne__(self, other):
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        return ~(self == other)

    def __hash__(self):
        return self._hash

    def __contains__(self, item) -> bool:
        """
        Return true if scalar item exists in this SpanArray.
        :param item: scalar Span value.
        :return: true if item exists in this SpanArray.
        """
        if isinstance(item, Span) and \
                item.begin == Span.NULL_OFFSET_VALUE:
            return Span.NULL_OFFSET_VALUE in self._begins
        return super().__contains__(item)

    def equals(self, other: "SpanArray"):
        """
        :param other: A second :class:`SpanArray`

        :return: ``True`` if both arrays have the same target texts (can be a
            different string object with the same contents) and the same spans
            in the same order.
        """
        if not isinstance(other, SpanArray):
            raise TypeError(f"equals() not defined for arguments of type "
                            f"{type(other)}")
        if self is other:
            return True

        # Check for cached result
        if id(other) in self._equivalent_arrays:
            cache_ix = self._equivalent_arrays.index(id(other))
        else:
            cache_ix = -1

        if (cache_ix >= 0
                and other.version == self._equiv_array_versions[cache_ix]):
            # Cached "equal" result
            return True
        elif (not np.array_equal(self.target_text, other.target_text)
              or not np.array_equal(self.begin, other.begin)
              or not np.array_equal(self.end, other.end)):
            # "Not equal" result from slow path
            if cache_ix >= 0:
                del self._equivalent_arrays[cache_ix]
                del self._equiv_array_versions[cache_ix]
            return False
        else:
            # If we get here, self and other are equal, and we had to expend
            # quite a bit of effort to figure that out.
            # Cache the result so we don't have to do that again.
            if cache_ix >= 0:
                self._equiv_array_versions[cache_ix] = other.version
            else:
                self._equivalent_arrays.append(id(other))
                self._equiv_array_versions.append(other.version)
            return True

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence[pd.api.extensions.ExtensionArray]
    ) -> pd.api.extensions.ExtensionArray:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        span_arrays = []  # type: List["SpanArray"]
        for tc in to_concat:
            if not isinstance(tc, SpanArray):
                raise ValueError(f"Attempted to concatenate a sequence containing a "
                                 f"non-SpanArray object via SpanArray._concat_same_type."
                                 f"  Types are: {[type(t) for t in to_concat]})")
            span_arrays.append(tc)

        string_table, text_ids_list = StringTable.merge_tables_and_ids(
            [s._string_table for s in span_arrays],
            [s._text_ids for s in span_arrays]
        )

        text_ids = np.concatenate(text_ids_list)
        begins = np.concatenate([a.begin for a in span_arrays])
        ends = np.concatenate([a.end for a in span_arrays])

        return SpanArray((string_table, text_ids), begins, ends)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(scalars, Span):
            scalars = [scalars]
        if isinstance(scalars, SpanArray):
            # Fast path for no-op
            scalars_as_span_array = scalars  # type: SpanArray
            if copy:
                return scalars_as_span_array.copy()
            else:
                return scalars_as_span_array

        begins = np.empty(len(scalars), dtype=int)
        ends = np.empty(len(scalars), dtype=int)
        target_texts = np.empty(len(scalars), dtype=object)
        i = 0
        for s in scalars:
            if not isinstance(s, Span):
                # TODO: Temporary fix for np.nan values, pandas-dev GH#38980
                try:
                    if np.isnan(s):  # May throw TypeError
                        s = _NULL_SPAN_SINGLETON
                    else:
                        raise TypeError()
                except TypeError:
                    raise ValueError(f"Can only convert a sequence of Span "
                                     f"objects to a SpanArray. Found an "
                                     f"object of type {type(s)}")
            begins[i] = s.begin
            ends[i] = s.end
            target_texts[i] = s.target_text
            i += 1
        string_table, text_ids = StringTable.merge_things(target_texts)
        return SpanArray((string_table, text_ids), begins, ends)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return cls._from_sequence(values)

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return self.astype(object), _NULL_SPAN_SINGLETON

    def isna(self) -> np.array:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return np.equal(self._begins, Span.NULL_OFFSET_VALUE)

    def copy(self) -> "SpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # StringTables are append-only, so shallow copying should be safe
        copy_str_table = self._string_table
        return SpanArray((copy_str_table, self._text_ids.copy()),
                         self.begin.copy(), self.end.copy())

    def take(
        self, indices: Sequence[int], allow_fill: bool = False,
        fill_value: Any = None
    ) -> "SpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if allow_fill:
            # From API docs: "[If allow_fill == True, then] negative values in
            # `indices` indicate missing values. These values are set to
            # `fill_value`.  Any other negative values raise a ``ValueError``."
            if fill_value is None or \
               (np.isscalar(fill_value) and np.isnan(fill_value)):
                fill_value = _NULL_SPAN_SINGLETON
            elif not isinstance(fill_value, Span):
                raise ValueError("Fill value must be Null, nan, or a Span "
                                 "(was {})".format(fill_value))
        else:
            # Dummy fill value to keep code below happy
            fill_value = _NULL_SPAN_SINGLETON

        # Pandas' internal implementation of take() does most of the heavy
        # lifting.
        begins = pd.api.extensions.take(
            self.begin, indices, allow_fill=allow_fill,
            fill_value=fill_value.begin
        )
        ends = pd.api.extensions.take(
            self.end, indices, allow_fill=allow_fill,
            fill_value=fill_value.end
        )
        text_ids = pd.api.extensions.take(
            self._text_ids, indices, allow_fill=allow_fill,
            fill_value=self._string_table.maybe_add_thing(fill_value.target_text)
        )

        # StringTables are append-only, so should be safe to share
        return SpanArray((self._string_table, text_ids), begins, ends)

    def __lt__(self, other):
        """
        Pandas-style array/series comparison function.

        :param other: Second operand of a Pandas "<" comparison with the series
        that wraps this TokenSpanArray.

        :return: Returns a boolean mask indicating which rows are less than
         `other`. span1 < span2 if span1.end <= span2.begin and both spans are over
         the same target text.
        """
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        elif not isinstance(other, (Span, SpanArray)):
            raise ValueError(f"'<' relationship not defined for {self} and {other} "
                             f"of types {type(self)} and {type(other)}.")
        else:
            offsets_mask = self.end <= other.begin
            text_mask = self.same_target_text(other)
            return np.logical_and(offsets_mask, text_mask)

    def __gt__(self, other):
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        if isinstance(other, (SpanArray, Span)):
            return other.__lt__(self)
        else:
            raise ValueError("'>' relationship not defined for {} and {} "
                             "of types {} and {}"
                             "".format(self, other, type(self), type(other)))

    def __le__(self, other):
        # TODO: Figure out what the semantics of this operation should be.
        raise NotImplementedError()

    def __ge__(self, other):
        # TODO: Figure out what the semantics of this operation should be.
        raise NotImplementedError()

    def _reduce(self, name, skipna=True, **kwargs):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if 0 == len(self):
            # Special case: Empty array
            # For all aggregates defined so far, we should return NA for this case.
            return _NULL_SPAN_SINGLETON

        if name == "sum":
            # Sum ==> combine, i.e. return the smallest span that contains all
            #         spans in the series
            if not self.is_single_document:
                raise ValueError(f"Sum of spans not defined for different target texts.")
            first_target_text = self.target_text[0]
            return Span(first_target_text, np.min(self.begin),
                        np.max(self.end))
        elif name == "first":
            return self[0]
            # return Span(first_target_text, self.begin[0], self.end[0])
        else:
            raise TypeError(f"'{name}' aggregation not supported on a series "
                            f"backed by a SpanArray")

    ####################################################
    # Methods that don't override the superclass go here

    @classmethod
    def make_array(cls, o) -> "SpanArray":
        """
        Make a :class:`SpanArray` object out of any of several types of input.

        :param o: a :class:`SpanArray` object represented as a :class:`pd.Series`,
            a list of :class:`Span` objects, or maybe just an actual :class:`SpanArray`
            (or :class:`TokenSpanArray`) object.

        :return: :class:`SpanArray` version of ``o``, which may be a pointer to ``o`` or
            one of its fields.
        """
        if isinstance(o, SpanArray):
            return o
        elif isinstance(o, pd.Series):
            return cls.make_array(o.values)
        elif isinstance(o, Sequence):
            return cls._from_sequence(o)
        elif isinstance(o, Iterable):
            return cls._from_sequence([e for e in o])

    @memoized_property
    def target_text(self) -> np.ndarray:
        """
        :return: "document" texts that the spans in this array reference, as opposed to
         the regions of these documents that the spans cover.
        """
        return self._string_table.ids_to_things(self._text_ids)

    @memoized_property
    def document_text(self) -> Union[str, None]:
        """
        :return: if all spans in this array cover the same document, text of that
         document.
         Raises a :class:`ValueError` if the array is empty or if the Spans in this
         array cover more than one document.
        """
        if len(self._text_ids) == 0:
            raise ValueError("An empty array has no document text")
        if not self.is_single_document:
            raise ValueError("Spans in array cover more than one document")
        else:
            # Look up first text directly so we don't materialize the target_text
            # property when it's not needed.
            return self._string_table.id_to_thing(self._text_ids[0])

    @memoized_property
    def is_single_document(self) -> bool:
        """
        :return: True if there is at least one span in the and every span is over the
         same target text.
        """
        # NOTE: For legacy reasons, this method is currently inconsistent with the method
        # by the same name in TokenSpanArray. TokenSpanArray.is_single_document() returns
        # True on an empty array, while SpanArray.is_single_document() returns false.
        if len(self) == 0:
            # If there are zero spans, then there are zero documents.
            return False
        elif self._string_table.num_things == 1:
            # Only one string; make sure that this array has a non-null value
            for b in self._begins:
                if b != Span.NULL_OFFSET_VALUE:
                    return True
            # All nulls --> zero spans
            return False
        else:
            # More than one string in the StringTable and at least one span.
            return self._is_single_document_slow_path()

    def _is_single_document_slow_path(self) -> bool:
        # Slow but reliable way to test whether everything in this SpanArray is from
        # the same document.
        # Checks whether every span has the same text ID.
        # Ignores NAs when making this comparison.

        # First we need to find the first text ID that is not NA
        first_text_id = None
        for b, t in zip(self._begins, self._text_ids):
            if b != Span.NULL_OFFSET_VALUE:
                first_text_id = t
                break
        if first_text_id is None:
            # Special case: All NAs --> Zero documents
            return False
        return not np.any(
            np.logical_and(
                # Row is not null...
                np.not_equal(self._begins, Span.NULL_OFFSET_VALUE),
                # ...and is over a different text than the first row's text ID
                np.not_equal(self._text_ids, first_text_id),
            )
        )

    def split_by_document(self) -> List["SpanArray"]:
        """
        :return: A list of slices of this `SpanArray` that cover single documents.
        """
        if self.is_single_document:
            return [self]
        slices = []
        for text_id in self._string_table.ids:
            mask = self._text_ids == text_id
            if np.any(mask):
                slices.append(self[mask])
        return slices

    @property
    def begin(self) -> np.ndarray:
        return self._begins

    @property
    def end(self) -> np.ndarray:
        return self._ends

    @property
    def version(self) -> int:
        """
        :return: Monotonically increasing version number that changes every time
        this array is modified. **NOTE:** This number might not change if a
        caller obtains a pointer to an internal array and modifies it.
        Callers who perform such modifications should call `increment_version()`
        """
        return self._version

    def increment_version(self):
        """
        Manually increase the version counter of this array to indicate that
        the array's contents have changed. Also invalidates any internal cached
        data derived from the array's state.
        """
        # Invalidate cached computation
        self._clear_cached_properties()

        self._equivalent_arrays = []
        self._equiv_array_versions = []

        # Increment the counter
        self._version += 1

    def as_tuples(self) -> np.ndarray:
        """
        :returns: (begin, end) pairs as an array of tuples
        """
        return np.concatenate(
            (self.begin.reshape((-1, 1)), self.end.reshape((-1, 1))),
            axis=1)

    @property
    def covered_text(self) -> np.ndarray:
        """
        :return: an array of the substrings of `target_text` corresponding to
        the spans in this array.
        """
        # TODO: Vectorized version of this
        texts = self.target_text
        # Need dtype=np.object so we can return nulls
        result = np.empty(len(self), dtype=object)
        for i in range(len(self)):
            if self._begins[i] == Span.NULL_OFFSET_VALUE:
                # Null value at this index
                result[i] = None
            elif texts[i] is None:
                # Null text and non-null begin/end. Shouldn't happen in normal use but
                # does occur in some parts of the Pandas regression suite.
                result[i] = None
            else:
                result[i] = texts[i][self._begins[i]:self._ends[i]]
        return result

    @memoized_property
    def normalized_covered_text(self) -> np.ndarray:
        """
        :return: A normalized version of the covered text of the spans in this
          array. Currently "normalized" means "lowercase".
        """
        # Currently we can't use np.char.lower directly because
        # self.covered_text needs to be an object array, not a numpy string
        # array, to allow for null values.
        covered_text = self.covered_text
        result = np.empty_like(covered_text)
        for i in range(len(result)):
            result[i] = None if covered_text[i] is None else covered_text[i].lower()
        return result

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

    def same_target_text(self, other: Union["SpanArray", Span]):
        """
        :param other: Either a single span or an array of spans of the same
            length as this one
        :return: Numpy array containing a boolean mask of all entries that
            have the same target text.
            Two spans with target text of None are considered to have the same
            target text.
        """
        if isinstance(other, Span):
            other_id = self._string_table.thing_to_id(other.target_text)
            return self._text_ids == other_id
        elif isinstance(other, SpanArray):
            other_ids = self._string_table.things_to_ids(other.target_text)
            return self._text_ids == other_ids
        else:
            raise TypeError(f"same_target_text not defined for input type "
                            f"{type(other)}")

    def overlaps(self, other: Union["SpanArray", Span]):
        """
        :param other: Either a single span or an array of spans of the same
            length as this one
        :return: Numpy array containing a boolean mask of all entries that
            overlap the corresponding element of `other`
        """
        if not isinstance(other, (Span, SpanArray)):
            raise TypeError(f"overlaps not defined for input type "
                            f"{type(other)}")

        # Replicate the logic in Span.overlaps() with boolean masks
        same_text_mask = self.same_target_text(other)
        exact_equal_mask = np.logical_and(self.begin == other.begin,
                                          self.end == other.end)
        begin_ge_end_mask = other.begin >= self.end
        end_le_begin_mask = other.end <= self.begin

        # (self.target_text == other.target_text) and (
        #   (self.begin == other.begin and self.end == other.end)
        #   or not (other.begin >= self.end or other.end <= self.begin)
        # )
        return (
            np.logical_and(
                same_text_mask,
                np.logical_or(
                    exact_equal_mask,
                    np.logical_not(
                        np.logical_or(begin_ge_end_mask,
                                      end_le_begin_mask)
                    )
                )
            )
        )

    def contains(self, other: Union["SpanArray", Span]):
        """
        :param other: Either a single span or an array of spans of the same
            length as this one
        :return: Numpy array containing a boolean mask of all entries that
            contain the corresponding element of `other`
        """
        if not isinstance(other, (Span, SpanArray)):
            raise TypeError(f"contains not defined for input type "
                            f"{type(other)}")

        # Replicate the logic in Span.contains() with boolean masks
        same_text_mask = self.same_target_text(other)
        begin_ge_begin_mask = other.begin >= self.begin
        end_le_end_mask = other.end <= self.end
        return (
            np.logical_and(
                same_text_mask,
                np.logical_and(begin_ge_begin_mask, end_le_end_mask)
            )
        )

    def _repr_html_(self) -> str:
        """
        HTML pretty-printing of a series of spans for Jupyter notebooks.
        """
        return jupyter.pretty_print_html(self, self._repr_html_show_offsets)

    @property
    def repr_html_show_offsets(self):
        """
        @returns: Whether the HTML/Jupyter notebook representation of this array will
         contain a table of span offsets in addition to the marked-up target text.
        """
        return self._repr_html_show_offsets

    @repr_html_show_offsets.setter
    def repr_html_show_offsets(self, show_offsets: bool):
        self._repr_html_show_offsets = show_offsets

    ##########################################
    # Keep private and protected methods here.

    @memoized_property
    def _hash(self):
        return hash((self.target_text.tobytes(), self._begins.tobytes(),
                     self._ends.tobytes()))

    # noinspection PyMethodMayBeStatic
    def _cached_property_names(self) -> List[str]:
        """
        :return: names of cached properties whose values are computed on demand
         and invalidated when the set of spans change.
        """
        return ["_hash", "is_single_document", "target_text",
                "normalized_covered_text", "document_text"]

    def _clear_cached_properties(self) -> None:
        """
        Remove cached values of memoized properties to reflect changes to the
        data on which they are based.
        """
        for n in self._cached_property_names():
            attr_name = "_{0}".format(n)
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def __arrow_array__(self, type=None):
        """
        Conversion of this Array to a pyarrow.ExtensionArray.
        :param type: Optional type passed to arrow for conversion, not used
        :return: pyarrow.ExtensionArray of type ArrowSpanType
        """
        from text_extensions_for_pandas.array.arrow_conversion import span_to_arrow
        return span_to_arrow(self)
