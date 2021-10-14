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
# token_span.py
#
# Part of text_extensions_for_pandas
#
# Pandas extensions to support columns of spans with token offsets.
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

from text_extensions_for_pandas.array.span import (
    Span,
    SpanArray,
    SpanDtype,
    SpanOpMixin,
)
# Internal imports
from text_extensions_for_pandas.array.token_table import TokenTable
from text_extensions_for_pandas.util import to_int_array


def _check_same_tokens(obj1, obj2):
    if isinstance(obj1, TokenSpan) and isinstance(obj2, TokenSpan):
        return obj1.tokens.equals(obj2.tokens)
    if not (isinstance(obj1, TokenSpanArray) or isinstance(obj2, TokenSpanArray)):
        raise TypeError(f"Expected some combination of TokenSpan and TokenSpanArray, "
                        f"but received {type(obj1)} and {type(obj2)}")

    same_tokens_mask = (
        obj1.same_tokens(obj2) if isinstance(obj1, TokenSpanArray)
        else obj2.same_tokens(obj1))
    if not np.all(same_tokens_mask):
        raise ValueError(
            f"TokenSpanArrays are over different sets of tokens "
            f"(got {obj1.tokens} and {obj2.tokens})\n"
            f"Comparison result: {same_tokens_mask}"
        )


class TokenSpanOpMixin(SpanOpMixin):
    """
    Mixin class to define common operations between TokenSpan and TokenSpanArray.
    """

    def __add__(self, other) -> Union[Span, "TokenSpan", SpanArray, "TokenSpanArray"]:
        """
        Add a pair of spans and/or span arrays.

        span1 + span2 == minimal span that covers both spans
        :param other: TokenSpan, Span, TokenSpanArray, or SpanArray
        :return: minimal span (or array of spans) that covers both inputs.
        """
        if isinstance(self, TokenSpan) and isinstance(other, TokenSpan):
            # TokenSpan + TokenSpan = TokenSpan
            _check_same_tokens(self, other)
            return TokenSpan(self.tokens, min(self.begin_token, other.begin_token),
                             max(self.end_token, other.end_token))
        elif isinstance(self, (TokenSpan, TokenSpanArray)) and \
                isinstance(other, (TokenSpan, TokenSpanArray)):
            # TokenSpanArray + TokenSpan* = TokenSpanArray
            _check_same_tokens(self, other)
            return TokenSpanArray(
                self.tokens,
                np.minimum(self.begin_token, other.begin_token),
                np.maximum(self.end_token, other.end_token))
        else:
            return super().__add__(other)


class TokenSpan(Span, TokenSpanOpMixin):
    """
    Python object representation of a single span with token offsets; that
    is, a single row of a `TokenSpanArray`.

    This class is also a subclass of `Span` and can return character-level
    information.

    An offset of `TokenSpan.NULL_OFFSET_VALUE` (currently -1) indicates
    "not a span" in the sense that NaN is "not a number".
    """

    def __init__(self, tokens: Any, begin_token: int, end_token: int):
        """
        :param tokens: Tokenization information about the document, including
        the target text. Must be a type that :func:`SpanArray.make_array()`
        can convert to a `SpanArray`.

        :param begin_token: Begin offset (inclusive) within the tokenized text,

        :param end_token: End offset; exclusive, one past the last token
        """
        tokens = SpanArray.make_array(tokens)
        if TokenSpan.NULL_OFFSET_VALUE != begin_token and begin_token < 0:
            raise ValueError(
                f"Begin token offset must be NULL_OFFSET_VALUE or "
                f"greater than zero (got {begin_token})"
            )
        if TokenSpan.NULL_OFFSET_VALUE != begin_token and end_token < begin_token:
            raise ValueError(
                f"End must be >= begin (got {begin_token} and " f"{end_token}"
            )
        if begin_token > len(tokens):
            raise ValueError(
                f"Begin token offset of {begin_token} larger than "
                f"number of tokens ({len(tokens)})"
            )
        if end_token > len(tokens) + 1:
            raise ValueError(
                f"End token offset of {end_token} larger than "
                f"number of tokens + 1 ({len(tokens)} + 1)"
            )
        if len(tokens) == 0 and begin_token != TokenSpan.NULL_OFFSET_VALUE:
            raise ValueError(
                f"Tried to create a non-null TokenSpan over an empty list of tokens."
            )
        if TokenSpan.NULL_OFFSET_VALUE == begin_token:
            if TokenSpan.NULL_OFFSET_VALUE != end_token:
                raise ValueError(
                    "Begin offset with special 'null' value {} "
                    "must be paired with an end offset of {}",
                    TokenSpan.NULL_OFFSET_VALUE,
                    TokenSpan.NULL_OFFSET_VALUE,
                )
            begin_char_off = end_char_off = Span.NULL_OFFSET_VALUE
        else:
            begin_char_off = tokens.begin[begin_token]
            end_char_off = (
                begin_char_off
                if begin_token == end_token
                else tokens.end[end_token - 1]
            )
        if len(tokens) == 0:
            doc_text = None
        elif not tokens.is_single_document:
            raise ValueError("Tokens must be from exactly one document.")
        else:
            doc_text = tokens.document_text

        super().__init__(doc_text, begin_char_off, end_char_off)
        self._tokens = tokens
        self._begin_token = begin_token
        self._end_token = end_token

    @classmethod
    def make_null(cls, tokens):
        """
        Convenience method for building null spans.
        :param tokens: Tokens of the target string
        :return: A null span over the indicated tokens
        """
        return TokenSpan(
            tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE
        )

    # Set this flag to True to use offets in tokens, not characters, in the
    # string representation of TokenSpans globally.
    USE_TOKEN_OFFSETS_IN_REPR = False

    def __repr__(self) -> str:
        if TokenSpan.NULL_OFFSET_VALUE == self._begin_token:
            return "NA"
        elif TokenSpan.USE_TOKEN_OFFSETS_IN_REPR:
            return "[{}, {}): '{}'".format(
                self.begin_token, self.end_token, textwrap.shorten(self.covered_text, 80)
            )
        else:
            return "[{}, {}): '{}'".format(
                self.begin, self.end, textwrap.shorten(self.covered_text, 80)
            )

    def __eq__(self, other):
        if isinstance(other, TokenSpan) and self.tokens.equals(other.tokens):
            return (
                self.begin_token == other.begin_token
                and self.end_token == other.end_token)
        else:
            # Different tokens, or no tokens, or not a span ==> Fall back on superclass
            return Span.__eq__(self, other)

    def __hash__(self):
        # Use superclass hash function so that hash and __eq__ are consistent
        return Span.__hash__(self)

    def __lt__(self, other):
        """
        span1 < span2 if span1.end <= span2.begin
        """
        if isinstance(other, TokenSpan):
            # Use token offsets when available
            return self.end_token <= other.begin_token
        else:
            return Span.__lt__(self, other)

    @property
    def tokens(self):
        return self._tokens

    @property
    def begin_token(self):
        return self._begin_token

    @property
    def end_token(self):
        return self._end_token


_EMPTY_SPAN_ARRAY_SINGLETON = SpanArray("", [], [])

_NULL_TOKEN_SPAN_SINGLETON = TokenSpan(_EMPTY_SPAN_ARRAY_SINGLETON,
                                       Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE)


@pd.api.extensions.register_extension_dtype
class TokenSpanDtype(SpanDtype):
    """
    Pandas datatype for a span that represents a range of tokens within a
    target string.
    """

    @property
    def type(self):
        # The type for a single row of a column of type TokenSpan
        return TokenSpan

    @property
    def name(self) -> str:
        """:return: A string representation of the dtype."""
        return "TokenSpanDtype"

    @property
    def na_value(self) -> object:
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        return _NULL_TOKEN_SPAN_SINGLETON

    @classmethod
    def construct_array_type(cls):
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        return TokenSpanArray

    def __from_arrow__(self, extension_array):
        """
        Convert the given extension array of type ArrowTokenSpanType to a
        TokenSpanArray.
        """
        from text_extensions_for_pandas.array.arrow_conversion import arrow_to_token_span
        return arrow_to_token_span(extension_array)


_NOT_A_DOCUMENT_TEXT = "This string is not the text of a document."
_EMPTY_INT_ARRAY = np.zeros(0, dtype=int)

# Singleton instance of the SpanArray value that corresponds to NA for tokens
# NULL_TOKENS_VALUE = SpanArray("", [], [])


class TokenSpanArray(SpanArray, TokenSpanOpMixin):
    """
    A Pandas :class:`ExtensionArray` that represents a column of token-based spans
    over a single target text.

    Spans are represented internally as ``[begin_token, end_token)`` intervals, where
    the properties ``begin_token`` and ``end_token`` are *token* offsets into the target
    text. As with the parent class :class:`SpanArray`, the properties ``begin`` and
    ``end`` of a :class:`TokenSpanArray` return *character* offsets.

    Null values are encoded with begin and end offsets of
    ``TokenSpan.NULL_OFFSET_VALUE``.

    Fields:

    * ``self._tokens``: Reference to the target string's tokens as a
      `SpanArray`. For now, references to different `SpanArray`
      objects are treated as different even if the arrays have the same
      contents.
    * ``self._begin_tokens``: Numpy array of integer offsets in tokens. An offset
      of TokenSpan.NULL_OFFSET_VALUE here indicates a null value.
    * ``self._end_tokens``: Numpy array of end offsets (1 + last token in span).
    """

    def __init__(self, tokens: Union[SpanArray, Sequence[SpanArray]],
                 begin_tokens: Union[pd.Series, np.ndarray, Sequence[int]] = None,
                 end_tokens: Union[pd.Series, np.ndarray, Sequence[int]] = None):

        """
        :param tokens: Character-level span information about the underlying
        tokens. Can be a single set of tokens, covering all spans, or a separate
        `SpanArray` pointer for every span.

        :param begin_tokens: Array of begin offsets measured in tokens
        :param end_tokens: Array of end offsets measured in tokens
        """
        # Superclass constructor expects values for things that the subclass doesn't
        # use.
        super().__init__(_NOT_A_DOCUMENT_TEXT, _EMPTY_INT_ARRAY, _EMPTY_INT_ARRAY)

        if not isinstance(begin_tokens, (pd.Series, np.ndarray, list)):
            raise TypeError(f"begin_tokens is of unsupported type {type(begin_tokens)}. "
                            f"Supported types are Series, ndarray and List[int].")
        if not isinstance(end_tokens, (pd.Series, np.ndarray, list)):
            raise TypeError(f"end_tokens is of unsupported type {type(end_tokens)}. "
                            f"Supported types are Series, ndarray and List[int].")

        if isinstance(tokens, SpanArray):
            if not tokens.is_single_document:
                raise ValueError(f"Token spans come from more than one document.")
            # Can't just pass a SpanArray to np.full() or np.array(), because Numpy will
            # interpret it as an array-like of Span values.
            tokens_array = np.empty(len(begin_tokens), dtype=object)
            for i in range(len(begin_tokens)):
                tokens_array[i] = tokens
            tokens = tokens_array
        elif isinstance(tokens, collections.abc.Sequence):
            if len(tokens) != len(begin_tokens):
                raise ValueError(f"Received {len(tokens)} arrays of tokens and "
                                 f"{len(begin_tokens)} begin offsets. "
                                 f"Lengths should be equal.")
            # Can't just pass a SpanArray to np.array(), because Numpy will interpret it
            # as an array-like of Span values.
            tokens_array = np.empty(len(begin_tokens), dtype=object)
            for i in range(len(begin_tokens)):
                tokens_array[i] = tokens[i]
            tokens = tokens_array
        elif isinstance(tokens, np.ndarray):
            if len(tokens) != len(begin_tokens):
                raise ValueError(f"Received {len(tokens)} arrays of tokens and "
                                 f"{len(begin_tokens)} begin offsets. "
                                 f"Lengths should be equal.")
            if (len(tokens) > 0
                    and tokens[0] is not None
                    and not isinstance(tokens[0], SpanArray)):
                raise TypeError(f"Tokens object for row 0 is of unexpected type "
                                f"{type(tokens[0])}. Type should be SpanArray.")
        else:
            raise TypeError(f"Expected SpanArray or list of SpanArray as tokens "
                            f"but got {type(tokens)}")

        self._tokens = tokens
        self._begin_tokens = to_int_array(begin_tokens)
        self._end_tokens = to_int_array(end_tokens)

    @staticmethod
    def from_char_offsets(tokens: Any) -> "TokenSpanArray":
        """
        Convenience factory method for wrapping the character-level spans of a
        series of tokens into single-token token-based spans.

        :param tokens: character-based offsets of the tokens, as any type that
         :func:`SpanArray.make_array` understands.

        :return: A :class:`TokenSpanArray` containing single-token spans for each of the
         tokens in ``tokens``.
        """
        begin_tokens = np.arange(len(tokens))
        tokens_array = SpanArray.make_array(tokens)
        return TokenSpanArray(tokens_array, begin_tokens, begin_tokens + 1)

    ##########################################
    # Overrides of superclass methods go here.

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        return TokenSpanDtype()

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
            data = self.to_numpy(dtype=dtype, copy=copy,
                                 na_value=_NULL_TOKEN_SPAN_SINGLETON)
        return data

    @property
    def nbytes(self) -> int:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        table, _ = TokenTable.merge_things(self.tokens)
        return (self._begin_tokens.nbytes + self._end_tokens.nbytes +
                table.nbytes())

    def __len__(self) -> int:
        return len(self._begin_tokens)

    def __getitem__(self, item) -> Union[TokenSpan, "TokenSpanArray"]:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(item, int):
            return TokenSpan(
                self.tokens[item], int(self._begin_tokens[item]),
                int(self._end_tokens[item])
            )
        else:
            # item not an int --> assume it's a numpy-compatible index
            item = check_array_indexer(self, item)
            return TokenSpanArray(
                self.tokens[item], self.begin_token[item], self.end_token[item]
            )

    def __setitem__(self, key: Union[int, np.ndarray, list, slice], value: Any) -> None:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """

        key = check_array_indexer(self, key)
        if isinstance(value, ABCSeries) and isinstance(value.dtype, SpanDtype):
            value = value.values

        if value is None or isinstance(value, Sequence) and len(value) == 0:
            self._begin_tokens[key] = TokenSpan.NULL_OFFSET_VALUE
            self._end_tokens[key] = TokenSpan.NULL_OFFSET_VALUE
        elif isinstance(value, TokenSpan):
            # Single input span --> one or more target positions
            self._begin_tokens[key] = value.begin_token
            self._end_tokens[key] = value.end_token

            # We'd like to do self._tokens[key] = value.tokens, but NumPy interprets
            # value.tokens as an array and gets very confused if you try that.
            mask = np.full(len(self._tokens), False, dtype=bool)
            mask[key] = True
            for i in range(len(self._tokens)):
                if mask[i]:
                    self._tokens[i] = value.tokens

        elif ((isinstance(key, slice) or
              (isinstance(key, np.ndarray) and is_bool_dtype(key.dtype)))
              and isinstance(value, TokenSpanArray)):
            # x spans -> x target positions
            self._tokens[key] = value.tokens
            self._begin_tokens[key] = value.begin_token
            self._end_tokens[key] = value.end_token
        elif (isinstance(key, np.ndarray) and len(value) > 0 and len(value) == len(key)
                and
                ((isinstance(value, Sequence) and isinstance(value[0], TokenSpan)) or
                 isinstance(value, TokenSpanArray))):
            for k, v in zip(key, value):
                self._tokens[k] = v.tokens
                self._begin_tokens[k] = v.begin_token
                self._end_tokens[k] = v.end_token
        else:
            raise ValueError(
                f"Attempted to set element of TokenSpanArray with "
                f"an object of type {type(value)}; current set of "
                f"allowed types is {(TokenSpan, TokenSpanArray)}"
            )

        self._clear_cached_properties()

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
        elif (isinstance(other, TokenSpanArray) and len(self) == len(other)
              and self.same_tokens(other)):
            return np.logical_and(
                self.begin_token == other.begin_token, self.end_token == other.end_token
            )
        else:
            # Different tokens, no tokens, unexpected type ==> fall back on superclass
            return SpanArray.__eq__(self, other)

    def __hash__(self):
        if self._hash is None:
            # Use superclass hash function so that hash() and == are consistent
            # across type.
            self._hash = SpanArray.__hash__(self)
        return self._hash

    def __contains__(self, item) -> bool:
        """
        Return true if scalar item exists in this TokenSpanArray.
        :param item: scalar TokenSpan value.
        :return: true if item exists in this TokenSpanArray.
        """
        if isinstance(item, TokenSpan) and \
                item.begin == TokenSpan.NULL_OFFSET_VALUE:
            return TokenSpan.NULL_OFFSET_VALUE in self._begin_tokens
        return super().__contains__(item)

    def __le__(self, other):
        # TODO: Figure out what the semantics of this operation should be.
        raise NotImplementedError()

    def __ge__(self, other):
        # TODO: Figure out what the semantics of this operation should be.
        raise NotImplementedError()

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence[pd.api.extensions.ExtensionArray]
    ) -> "TokenSpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if len(to_concat) == 0:
            raise ValueError("Can't concatenate zero TokenSpanArrays")
        arrays_to_concat = []  # type: List[TokenSpanArray]
        for c in to_concat:
            if not isinstance(c, TokenSpanArray):
                raise TypeError(f"Tried to concatenate {type(c)} to TokenSpanArray")
            arrays_to_concat.append(c)

        tokens = np.concatenate([a.tokens for a in arrays_to_concat])
        begin_tokens = np.concatenate([a.begin_token for a in arrays_to_concat])
        end_tokens = np.concatenate([a.end_token for a in arrays_to_concat])

        return cls(tokens, begin_tokens, end_tokens)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # Because we don't currently override the factorize() class method, the
        # "values" input to _from_factorized is a ndarray of TokenSpan objects.
        # TODO: Faster implementation of factorize/_from_factorized
        # Can't pass SpanArrays to np.array() because SpanArrays are array-like.
        begin_tokens = np.array([v.begin_token for v in values], dtype=np.int32)
        end_tokens = np.array([v.end_token for v in values], dtype=np.int32)
        tokens = np.empty(len(begin_tokens), dtype=object)
        i = 0
        for v in values:
            tokens[i] = v.tokens
            i += 1
        return cls(tokens, begin_tokens, end_tokens)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(scalars, TokenSpan):
            scalars = [scalars]

        # noinspection PyTypeChecker
        tokens = np.empty(len(scalars), object)
        begin_tokens = np.empty(len(scalars), np.int32)
        end_tokens = np.empty(len(scalars), np.int32)

        i = 0
        for s in scalars:
            if not isinstance(s, TokenSpan):
                # TODO: Temporary fix for np.nan values, pandas-dev GH#38980
                if np.isnan(s):
                    s = _NULL_TOKEN_SPAN_SINGLETON
                else:
                    raise ValueError(
                        f"Can only convert a sequence of TokenSpan "
                        f"objects to a TokenSpanArray. Found an "
                        f"object of type {type(s)}"
                    )
            tokens[i] = s.tokens
            begin_tokens[i] = s.begin_token
            end_tokens[i] = s.end_token
            i += 1
        return TokenSpanArray(tokens, begin_tokens, end_tokens)

    def isna(self) -> np.array:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # isna() of an ExtensionArray must return a copy that the caller can scribble on.
        return self.nulls_mask.copy()

    def copy(self) -> "TokenSpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        ret = TokenSpanArray(
            self.tokens, self.begin_token.copy(), self.end_token.copy()
        )
        # TODO: Copy cached properties
        return ret

    def take(
        self, indices: Sequence[int], allow_fill: bool = False, fill_value: Any = None
    ) -> "TokenSpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # From API docs: "[If allow_fill == True, then] negative values in
        # `indices` indicate missing values. These values are set to
        # `fill_value`.
        if fill_value is None or \
                (np.isscalar(fill_value) and np.isnan(fill_value)):
            # Replace with a "nan span"
            fill_value = _NULL_TOKEN_SPAN_SINGLETON
        elif not isinstance(fill_value, TokenSpan):
            raise ValueError(
                "Fill value must be Null, nan, or a TokenSpan "
                "(was {})".format(fill_value)
            )

        # Pandas' internal implementation of take() does most of the heavy
        # lifting.
        tokens = pd.api.extensions.take(
            self.tokens,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value.tokens,
        )
        begin_tokens = pd.api.extensions.take(
            self.begin_token,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value.begin_token,
        )
        end_tokens = pd.api.extensions.take(
            self.end_token,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value.end_token,
        )

        return TokenSpanArray(
            tokens,
            begin_tokens,
            end_tokens
        )

    ####################################################
    # Methods that don't override the superclass go here

    @classmethod
    def make_array(cls, o) -> "TokenSpanArray":
        """
        Make a :class:`TokenSpanArray` object out of any of several types of input.

        :param o: a :class:`TokenSpanArray` object represented as a :class:`pd.Series`,
            a list of :class:`TokenSpan` objects, or an actual :class:`TokenSpanArray`
            object.

        :return: :class:`TokenSpanArray` version of ``o``, which may be a pointer to ``o`` or
            one of its fields.
        """
        if isinstance(o, TokenSpanArray):
            return o
        elif isinstance(o, pd.Series):
            return cls.make_array(o.values)
        elif isinstance(o, Sequence):
            return cls._from_sequence(o)
        elif isinstance(o, Iterable):
            return cls._from_sequence([e for e in o])

    @classmethod
    def align_to_tokens(cls, tokens: Any, spans: Any):
        """
        Align a set of character or token-based spans to a specified
        tokenization, producing a `TokenSpanArray` of token-based spans.

        :param tokens: The tokens to align to, as any type that
         :func:`SpanArray.make_array` accepts.
        :param spans: The spans to align. These spans must all target the same text
         as ``tokens``.

        :return: An array of :class:`TokenSpan` objects aligned to the tokens of
            ``tokens``.
            Raises :class:`ValueError` if any of the spans in ``spans`` doesn't start and
            end on a token boundary.
        """
        tokens = SpanArray.make_array(tokens)
        spans = SpanArray.make_array(spans)

        if not tokens.is_single_document:
            raise ValueError(f"Tokens cover more than one document (tokens are {tokens})")
        if not spans.is_single_document:
            raise ValueError(f"Spans cover more than one document (spans are {spans})")

        # Create and join temporary dataframes
        tokens_df = pd.DataFrame({
            "token_index": np.arange(len(tokens)),
            "token_begin": tokens.begin,
            "token_end": tokens.end
        })
        spans_df = pd.DataFrame({
            "span_index": np.arange(len(spans)),
            "span_begin": spans.begin,
            "span_end": spans.end
        })

        # Ignore zero-length tokens
        # TODO: Is this the right thing to do?
        tokens_df = tokens_df[tokens_df["token_begin"] != tokens_df["token_end"]]

        begin_matches = pd.merge(tokens_df, spans_df,
                                 left_on="token_begin",
                                 right_on="span_begin",
                                 how="right", indicator=True)

        mismatched = begin_matches[begin_matches["_merge"] == "right_only"]
        if len(mismatched.index) > 0:
            raise ValueError(
                f"The following span(s) did not align with the begin offset\n"
                f"of any token:\n"
                f"{mismatched[['span_index', 'span_begin', 'span_end']]}")

        end_matches = pd.merge(tokens_df, spans_df,
                               left_on="token_end",
                               right_on="span_end",
                               how="right", indicator=True)

        mismatched = end_matches[end_matches["_merge"] == "right_only"]
        if len(mismatched.index) > 0:
            raise ValueError(
                f"The following span(s) did not align with the end offset\n"
                f"of any token:\n"
                f"{mismatched[['span_index', 'span_begin', 'span_end']]}")

        # Join on span index to get (begin, end) pairs.
        begins_and_ends = pd.merge(
            begin_matches[["token_index", "span_index"]],
            end_matches[["token_index", "span_index"]],
            on="span_index", suffixes=("_begin", "_end"),
            sort=True)

        return TokenSpanArray(tokens,
                              begins_and_ends["token_index_begin"],
                              begins_and_ends["token_index_end"] + 1)

    @property
    def tokens(self) -> np.ndarray:
        """
        :return: The tokens over which each TokenSpan in this array are defined as
         an ndarray of object.
        """
        return self._tokens

    @memoized_property
    def target_text(self) -> np.ndarray:
        """
        :return: "document" texts that the spans in this array reference, as opposed to
         the regions of these documents that the spans cover.
        """
        # Note that this property overrides the eponymous property in SpanArray
        texts = [
            None if self.nulls_mask[i]
            else self.tokens[i].document_text
            for i in range(len(self))
        ]
        return np.array(texts, dtype=object)

    @memoized_property
    def document_text(self) -> Union[str, None]:
        """
        :return: if all spans in this array cover the same document, text of that
             document.
             Raises a :class:`ValueError` if the array is empty or if the Spans in this
             array cover more than one document.
        """
        # Checks for zero-length array and multiple docs are in document_tokens()
        return self.document_tokens.document_text

    @memoized_property
    def document_tokens(self) -> Union[SpanArray, None]:
        """
        :return: if all spans in this array cover the same tokenization of a single
         document, tokens of that document.
         Raises a `ValueError` if the array is empty or if the Spans in this
         array cover more than one document.
        """
        if len(self.tokens) == 0:
            raise ValueError("An empty array has no document tokens")
        elif not self.is_single_document:
            raise ValueError("Spans in array cover more than one document")
        else:
            return self.tokens[0]

    @memoized_property
    def nulls_mask(self) -> np.ndarray:
        """
        :return: A boolean mask indicating which rows are nulls
        """
        return self._begin_tokens == TokenSpan.NULL_OFFSET_VALUE

    @memoized_property
    def begin(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span begins.
        """
        result = np.empty_like(self.begin_token, dtype=np.int32)
        for i in range(len(self)):
            begin_token_ix = self.begin_token[i]
            if begin_token_ix == TokenSpan.NULL_OFFSET_VALUE:
                result[i] = Span.NULL_OFFSET_VALUE
            else:
                result[i] = self.tokens[i].begin[begin_token_ix]

        return result

    @memoized_property
    def end(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span ends.
        """
        # Start out with the end of the last token in each span.
        result = np.empty_like(self.begin_token, dtype=np.int32)
        for i in range(len(self)):
            begin_token_ix = self.begin_token[i]
            end_token_ix = self.end_token[i]
            if begin_token_ix == TokenSpan.NULL_OFFSET_VALUE:
                result[i] = Span.NULL_OFFSET_VALUE
            elif begin_token_ix == end_token_ix:
                # Zero-length span
                result[i] = self.begin[i]
            else:
                result[i] = self.tokens[i].end[end_token_ix - 1]

        return result

    @property
    def begin_token(self) -> np.ndarray:
        """
        :return: Token offsets of the span begins; that is, the index of the
        first token in each span.
        """
        return self._begin_tokens

    @property
    def end_token(self) -> np.ndarray:
        """
        :return: Token offsets of the span ends. That is, 1 + last token
        present in the span, for each span in the column.
        """
        return self._end_tokens

    def as_tuples(self) -> np.ndarray:
        """
        Returns (begin, end) pairs as an array of tuples
        """
        return np.concatenate(
            (self.begin.reshape((-1, 1)), self.end.reshape((-1, 1))), axis=1
        )

    def increment_version(self):
        """
        Override parent class's version of this function to also clear out data cached
        in the subclass.
        """
        super().increment_version()

    @memoized_property
    def covered_text(self) -> np.ndarray:
        """
        Returns an array of the substrings of `target_text` corresponding to
        the spans in this array.
        """
        texts = [
            None if self.nulls_mask[i]
            else self.target_text[i][self.begin[i]:self.end[i]]
            for i in range(len(self))
        ]
        return np.array(texts, dtype=object)

    def as_frame(self) -> pd.DataFrame:
        """
        Returns a dataframe representation of this column based on Python
        atomic types.
        """
        return pd.DataFrame(
            {
                "begin": self.begin,
                "end": self.end,
                "begin_token": self.begin_token,
                "end_token": self.end_token,
                "covered_text": self.covered_text,
            }
        )

    def same_target_text(self, other: Union["SpanArray", Span]):
        """
        :param other: Either a single span or an array of spans of the same
            length as this one
        :return: Numpy array containing a boolean mask of all entries that
            have the same target text.
            Two spans with target text of None are considered to have the same
            target text.
        """
        if isinstance(other, (Span, SpanArray)):
            return self.target_text == other.target_text
        else:
            raise TypeError(f"same_target_text not defined for input type "
                            f"{type(other)}")

    def same_tokens(self, other: Union["TokenSpanArray", TokenSpan]):
        """
        :param other: Either a single span or an array of spans of the same
            length as this one. Must be token-based.
        :return: Numpy array containing a boolean mask of all entries that
            are over the same tokenization of the same target text.
            Two spans with target text of None are considered to have the same
            target text.
        """
        if not isinstance(other, (TokenSpan, TokenSpanArray)):
            raise TypeError(f"same_tokens not defined for input type "
                            f"{type(other)}")

        if self.is_single_tokenization:
            # Fast path for common case of one set of tokens
            other_tokens = (other.tokens if isinstance(other, TokenSpan)
                            else other.document_tokens)
            return self.document_tokens.equals(other_tokens)

        # Slow path: Compare each element.
        if isinstance(other, TokenSpan):
            return np.array([t.equals(other.tokens) for t in self.tokens], dtype=bool)
        else:  # isinstance(other, TokenSpanArray)
            return np.array([self.tokens[i].equals(other.tokens[i])
                             for i in range(len(self.tokens))], dtype=bool)

    @memoized_property
    def is_single_document(self) -> bool:
        """
        :return: True if every span in this array is over the same target text
         or if there are zero spans in this array.
        """
        # NOTE: For legacy reasons, this method is currently inconsistent with the method
        # by the same name in SpanArray. TokenSpanArray.is_single_document() returns
        # True on an empty array, while SpanArray.is_single_document() returns False.
        if len(self) == 0:
            # If there are zero spans, we consider there to be one document with the
            # document text being whatever is the document text for our tokens.
            return True
        else:
            # More than one tokenization and at least one span. Check whether
            # every span has the same text.

            # Find the first span that is not NA
            first_target_text = None
            for b, t in zip(self._begin_tokens, self.target_text):
                if b != Span.NULL_OFFSET_VALUE:
                    first_target_text = t
                    break
            if first_target_text is None:
                # Special case: All NAs --> Zero documents
                return True
            return not np.any(np.logical_and(
                # Row is not null...
                np.not_equal(self._begin_tokens, Span.NULL_OFFSET_VALUE),
                # ...and is over a different text than the first row's text ID
                np.not_equal(self.target_text, first_target_text)))

    def split_by_document(self) -> List["SpanArray"]:
        """
        :return: A list of slices of this `SpanArray` that cover single documents.
        """
        if self.is_single_document:
            return [self]

        # For now, treat each tokenization as a different document to avoid O(n^2)
        # behavior.
        # TODO: Consider a more in-depth comparison to capture mixtures of different
        #  tokenizations of the same document.
        token_table, token_ids = TokenTable.merge_things(self.tokens)
        result = []
        for tokens_id in token_table.ids:
            mask = token_ids == tokens_id
            if np.any(mask):
                result.append(self[mask])
        return result

    @memoized_property
    def is_single_tokenization(self) -> bool:
        """
        :return: True if every span in this array is over the same tokenization
         of the same target text or if there are zero spans in this array.
        """
        if len(self) == 0:
            # If there are zero spans, we consider there to be one document with the
            # document text being whatever is the first element of the StringTable.
            return True
        else:
            first_t = self.tokens[0]
            for t in self.tokens:
                if not t.equals(first_t):
                    return False
            return True

    ##########################################
    # Keep private and protected methods here.

    def _cached_property_names(self) -> List[str]:
        """
        :return: names of cached properties whose values are computed on demand
         and invalidated when the set of spans change.
        """
        # Superclass has its own list.
        return super()._cached_property_names() + [
            "nulls_mask", "have_nulls", "begin", "end", "target_text",
            "covered_text", "document_tokens"
            ]

    def __arrow_array__(self, type=None):
        """
        Conversion of this Array to a pyarrow.ExtensionArray.
        :param type: Optional type passed to arrow for conversion, not used
        :return: pyarrow.ExtensionArray of type ArrowTokenSpanType
        """
        from text_extensions_for_pandas.array.arrow_conversion import token_span_to_arrow
        return token_span_to_arrow(self)
