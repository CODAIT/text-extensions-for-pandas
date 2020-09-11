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
import textwrap
from typing import *

import numpy as np
import pandas as pd
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries
from pandas.core.indexers import check_array_indexer
from pandas.api.types import is_bool_dtype
from memoized_property import memoized_property

# Internal imports
from text_extensions_for_pandas.array.span import (
    Span,
    SpanArray,
    SpanDtype,
    SpanOpMixin
)


def _check_same_tokens(array1, array2):
    if not array1.tokens.equals(array2.tokens):
        raise ValueError(
            f"TokenSpanArrays are over different sets of tokens "
            f"(got {array1.tokens} and {array2.tokens})"
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
            return TokenSpanArray(self.tokens,
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

    def __init__(self, tokens: SpanArray, begin_token: int, end_token: int):
        """
        :param tokens: Tokenization information about the document, including
        the target text.

        :param begin_token: Begin offset (inclusive) within the tokenized text,

        :param end_token: End offset; exclusive, one past the last token
        """
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
                f"End token offset of {begin_token} larger than "
                f"number of tokens + 1 ({len(tokens)} + 1)"
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
        super().__init__(tokens.target_text, begin_char_off, end_char_off)
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


class TokenSpanArray(SpanArray, TokenSpanOpMixin):
    """
    A Pandas `ExtensionArray` that represents a column of token-based spans
    over a single target text.

    Spans are represented internaly as `[begin_token, end_token)` intervals, where
    the properties `begin_token` and `end_token` are *token* offsets into the target
    text. As with the parent class `SpanArray`, the properties `begin` and `end`
    of a `TokenSpanArray` return *character* offsets.

    Null values are encoded with begin and end offsets of
    `TokenSpan.NULL_OFFSET_VALUE`.

    Fields:
    * `self._tokens`: Reference to the target string's tokens as a
        `SpanArray`. For now, references to different `SpanArray`
        objects are treated as different even if the arrays have the same
        contents.
    * `self._begin_tokens`: Numpy array of integer offsets in tokens. An offset
       of TokenSpan.NULL_OFFSET_VALUE here indicates a null value.
    * `self._end_tokens`: Numpy array of end offsets (1 + last token in span).
    """

    @staticmethod
    def from_char_offsets(tokens: SpanArray) -> "TokenSpanArray":
        """
        Convenience factory method for wrapping the character-level spans of a
        series of tokens into single-token token-based spans.

        :param tokens: character-based offsets of the tokens

        :return: A TokenSpanArray containing single-token spans for each of the
        tokens in `tokens`.
        """
        begin_tokens = np.arange(len(tokens))
        return TokenSpanArray(tokens, begin_tokens, begin_tokens + 1)

    def __init__(
        self,
        tokens: SpanArray,
        begin_tokens: Union[pd.Series, np.ndarray, Sequence[int]] = None,
        end_tokens: Union[pd.Series, np.ndarray, Sequence[int]] = None,
    ):
        """
        :param tokens: Character-level span information about the underlying
        tokens.

        :param begin_tokens: Array of begin offsets measured in tokens
        :param end_tokens: Array of end offsets measured in tokens
        """
        if not isinstance(tokens, SpanArray):
            raise TypeError(f"Expected SpanArray as tokens but got {type(tokens)}")
        if not isinstance(begin_tokens, (pd.Series, np.ndarray, list)):
            raise TypeError(f"begin_tokens is of unsupported type {type(begin_tokens)}. "
                            f"Supported types are Series, ndarray and List[int].")
        if not isinstance(end_tokens, (pd.Series, np.ndarray, list)):
            raise TypeError(f"end_tokens is of unsupported type {type(end_tokens)}. "
                            f"Supported types are Series, ndarray and List[int].")

        super().__init__(tokens.target_text, tokens.begin, tokens.end)

        begin_tokens = (
            np.array(begin_tokens) if not isinstance(begin_tokens, np.ndarray)
            else begin_tokens
        )
        end_tokens = (
            np.array(end_tokens) if not isinstance(end_tokens, np.ndarray)
            else end_tokens
        )

        self._begin_tokens = begin_tokens  # Type: np.ndarray
        self._end_tokens = end_tokens  # Type: np.ndarray

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
            return dtype.construct_array_type()._from_sequence(self, copy=False)
        else:
            na_value = TokenSpan(
                self.tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE
            )
            data = self.to_numpy(dtype=dtype, copy=copy, na_value=na_value)
        return data

    @property
    def nbytes(self) -> int:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return self._begin_tokens.nbytes + self._end_tokens.nbytes + super().nbytes

    def __len__(self) -> int:
        return len(self._begin_tokens)

    def __getitem__(self, item) -> Union[TokenSpan, "TokenSpanArray"]:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(item, int):
            return TokenSpan(
                self.tokens, int(self._begin_tokens[item]), int(self._end_tokens[item])
            )
        else:
            # item not an int --> assume it's a numpy-compatible index
            item = check_array_indexer(self, item)
            return TokenSpanArray(
                self.tokens, self.begin_token[item], self.end_token[item]
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
        elif isinstance(value, TokenSpan) or \
                ((isinstance(key, slice) or
                  (isinstance(key, np.ndarray) and is_bool_dtype(key.dtype))) and
                 isinstance(value, SpanArray)):
            self._begin_tokens[key] = value.begin_token
            self._end_tokens[key] = value.end_token
        elif isinstance(key, np.ndarray) and len(value) > 0 and len(value) == len(key) and \
                ((isinstance(value, Sequence) and isinstance(value[0], TokenSpan)) or
                 isinstance(value, TokenSpanArray)):
            for k, v in zip(key, value):
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
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndexClass)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        if isinstance(other, TokenSpan) and self.tokens.equals(other.tokens):
            mask = np.full(len(self), True, dtype=np.bool)
            mask[self.begin_token != other.begin_token] = False
            mask[self.end_token != other.end_token] = False
            return mask
        elif isinstance(other, TokenSpanArray) and self.tokens.equals(other.tokens):
            if len(self) != len(other):
                raise ValueError(
                    "Can't compare arrays of differing lengths "
                    "{} and {}".format(len(self), len(other))
                )
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
        # Require exact object equality of the tokens for now.
        tokens = to_concat[0].tokens
        for c in to_concat:
            if not c.tokens.equals(tokens):
                raise ValueError(
                    "Can only concatenate spans that are over " "the same set of tokens"
                )
        begin_tokens = np.concatenate([a.begin_token for a in to_concat])
        end_tokens = np.concatenate([a.end_token for a in to_concat])
        return TokenSpanArray(tokens, begin_tokens, end_tokens)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # Because we don't currently override the factorize() class method, the
        # "values" input to _from_factorized is a ndarray of TokenSpan objects.
        # TODO: Faster implementation of factorize/_from_factorized
        begin_tokens = np.array([v.begin_token for v in values], dtype=np.int)
        end_tokens = np.array([v.end_token for v in values], dtype=np.int)
        return TokenSpanArray(original.tokens, begin_tokens, end_tokens)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        tokens = None
        if isinstance(scalars, Span):
            scalars = [scalars]
        if isinstance(scalars, TokenSpanArray):
            tokens = scalars.tokens
        begin_tokens = np.full(len(scalars), TokenSpan.NULL_OFFSET_VALUE, np.int)
        end_tokens = np.full(len(scalars), TokenSpan.NULL_OFFSET_VALUE, np.int)
        i = 0
        for s in scalars:
            if not isinstance(s, TokenSpan):
                raise ValueError(
                    f"Can only convert a sequence of TokenSpan "
                    f"objects to a TokenSpanArray. Found an "
                    f"object of type {type(s)}"
                )
            if tokens is None:
                tokens = s.tokens
            if not s.tokens.equals(tokens):
                raise ValueError(
                    f"Mixing different token sets is not currently "
                    f"supported. Received two token sets:\n"
                    f"{tokens}\nand\n{s.tokens}"
                )
            begin_tokens[i] = s.begin_token
            end_tokens[i] = s.end_token
            i += 1
        return TokenSpanArray(tokens, begin_tokens, end_tokens)

    def isna(self) -> np.array:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return self.nulls_mask

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
            (np.isscalar(fill_value) and np.math.isnan(fill_value)):
            # Replace with a "nan span"
            fill_value = TokenSpan(
                self.tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE
            )
        elif not isinstance(fill_value, TokenSpan):
            raise ValueError(
                "Fill value must be Null, nan, or a TokenSpan "
                "(was {})".format(fill_value)
            )

        # Pandas' internal implementation of take() does most of the heavy
        # lifting.
        begins = pd.api.extensions.take(
            self.begin_token,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value.begin_token,
        )
        ends = pd.api.extensions.take(
            self.end_token,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value.end_token,
        )
        return TokenSpanArray(self.tokens, begins, ends)

    def __lt__(self, other) -> np.ndarray:
        """
        Pandas/Numpy-style array/series comparison function.

        :param other: Second operand of a Pandas "<" comparison with the series
        that wraps this TokenSpanArray.

        :return: Returns a boolean mask indicating which rows are less than
         `other`. span1 < span2 if span1.end <= span2.begin.
        """
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndexClass)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        if isinstance(other, (TokenSpanArray, TokenSpan)):
            # Use token offsets when available.
            return self.end_token <= other.begin_token
        elif isinstance(other, (SpanArray, Span)):
            return self.end <= other.begin
        else:
            raise ValueError(
                "'<' relationship not defined for {} and {} "
                "of types {} and {}"
                "".format(self, other, type(self), type(other))
            )

    def __gt__(self, other) -> np.ndarray:
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndexClass)):
            # Rely on pandas to unbox and dispatch to us.
            return NotImplemented
        if isinstance(other, (TokenSpanArray, TokenSpan)):
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
        if name == "sum":
            # Sum ==> combine, i.e. return the smallest span that contains all
            #         spans in the series
            return TokenSpan(
                self.tokens, np.min(self.begin_token), np.max(self.end_token)
            )
        else:
            raise TypeError(
                f"'{name}' aggregation not supported on a series "
                f"backed by a TokenSpanArray"
            )

    ####################################################
    # Methods that don't override the superclass go here

    @classmethod
    def make_array(cls, o) -> "TokenSpanArray":
        """
        Make a `TokenSpanArray` object out of any of several types of input.

        :param o: a TokenSpanArray object represented as a `pd.Series`, a list
        of `TokenSpan` objects, or maybe just an actual `TokenSpanArray` object.

        :return: TokenSpanArray version of `o`, which may be a pointer to `o` or
        one of its fields.
        """
        if isinstance(o, TokenSpanArray):
            return o
        elif isinstance(o, pd.Series):
            return cls.make_array(o.values)
        elif isinstance(o, Iterable):
            return cls._from_sequence(o)

    @classmethod
    def align_to_tokens(cls, tokens: Any, spans: Any):
        """
        Align a set of character or token-based spans to a specified
        tokenization, producing a `TokenSpanArray` of token-based spans.

        :param tokens: The tokens to align to, as any type that
         `SpanArray.make_array()` accepts.
        :param spans: The spans to align.
        :return: An array of `TokenSpan`s aligned to the tokens of `tokens`.
         Raises `ValueError` if any of the spans in `spans` doesn't start and
         end on a token boundary.
        """
        tokens = SpanArray.make_array(tokens)
        spans = SpanArray.make_array(spans)

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
    def tokens(self) -> SpanArray:
        return SpanArray(self._text, self._begins, self._ends)

    @property
    def target_text(self) -> str:
        """
        :return: the common "document" text that the spans in this array
        reference.
        """
        return self._text

    @memoized_property
    def nulls_mask(self) -> np.ndarray:
        """
        :return: A boolean mask indicating which rows are nulls
        """
        return self._begin_tokens == TokenSpan.NULL_OFFSET_VALUE

    @memoized_property
    def have_nulls(self) -> bool:
        """
        :return: True if this column contains one or more nulls
        """
        return np.any(self.nulls_mask)

    @memoized_property
    def begin(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span begins.
        """
        result = self._begins[self.begin_token]
        # Correct for null values
        result[self.nulls_mask] = TokenSpan.NULL_OFFSET_VALUE
        return result

    @memoized_property
    def end(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span ends.
        """
        # Start out with the end of the last token in each span.
        result = self._ends[self.end_token - 1]
        # Replace end offset with begin offset wherever the length in tokens
        # is zero.
        mask = self.end_token == self.begin_token
        result[mask] = self.begin[mask]
        # Correct for null values
        result[self.nulls_mask] = TokenSpan.NULL_OFFSET_VALUE
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

    @property
    def covered_text(self) -> np.ndarray:
        """
        Returns an array of the substrings of `target_text` corresponding to
        the spans in this array.
        """
        # TODO: Vectorized version of this
        text = self.target_text
        # Need dtype=np.object so we can return nulls
        result = np.zeros(len(self), dtype=np.object)
        for i in range(len(self)):
            if self._begin_tokens[i] == TokenSpan.NULL_OFFSET_VALUE:
                # Null value at this index
                result[i] = None
            else:
                result[i] = text[self.begin[i] : self.end[i]]
        return result

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

    ##########################################
    # Keep private and protected methods here.

    def _clear_cached_properties(self) -> None:
        """
        Remove cached values of memoized properties to reflect changes to the
        data on which they are based.
        """
        # TODO: Figure out how to generate this list automatically
        property_names = ["nulls_mask", "have_nulls", "begin", "end"]
        for n in property_names:
            attr_name = "_{0}".format(n)
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self._hash = None

    def __arrow_array__(self, type=None):
        """
        Conversion of this Array to a pyarrow.ExtensionArray.
        :param type: Optional type passed to arrow for conversion, not used
        :return: pyarrow.ExtensionArray of type ArrowTokenSpanType
        """
        from text_extensions_for_pandas.array.arrow_conversion import token_span_to_arrow
        return token_span_to_arrow(self)
