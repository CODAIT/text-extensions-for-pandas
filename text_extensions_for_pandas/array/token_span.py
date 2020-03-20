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

from typing import *

import numpy as np
import pandas as pd
from memoized_property import memoized_property

# Internal imports
import text_extensions_for_pandas.util as util
from text_extensions_for_pandas.array.char_span import (
    CharSpan,
    CharSpanArray,
    CharSpanType,
)


class TokenSpan(CharSpan):
    """
    Python object representation of a single span with token offsets; that
    is, a single row of a `TokenSpanArray`.

    This class is also a subclass of `CharSpan` and can return character-level
    information.

    An offset of `TokenSpan.NULL_OFFSET_VALUE` (currently -1) indicates
    "not a span" in the sense that NaN is "not a number".
    """

    def __init__(self, tokens: CharSpanArray, begin_token: int, end_token: int):
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
            begin_char_off = end_char_off = CharSpan.NULL_OFFSET_VALUE
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
            return "Nil"
        elif TokenSpan.USE_TOKEN_OFFSETS_IN_REPR:
            return "[{}, {}): '{}'".format(
                self.begin_token, self.end_token, util.truncate_str(self.covered_text)
            )
        else:
            return "[{}, {}): '{}'".format(
                self.begin, self.end, util.truncate_str(self.covered_text)
            )

    def __eq__(self, other):
        return (
            isinstance(other, TokenSpan)
            and self.tokens.equals(other.tokens)
            and self.begin_token == other.begin_token
            and self.end_token == other.end_token
        )

    def __hash__(self):
        result = hash((self.tokens, self.begin_token, self.end_token))
        return result

    def __lt__(self, other):
        """
        span1 < span2 if span1.end <= span2.begin
        """
        if isinstance(other, TokenSpan):
            # Use token offsets when available
            return self.end_token <= other.begin_token
        else:
            return CharSpan.__lt__(self, other)

    def __add__(self, other):
        """
        span1 + span2 == minimal span that covers both spans
        :param other: Other span to add to this one. Currently constrained to
            be a single TokenSpan. Eventually this argument will permit
            CharSpan, CharSpanArray, and TokenSpanArray
        :return: minimal span that covers both spans
        """
        if isinstance(other, TokenSpan):
            if not self.tokens.equals(other.tokens):
                raise ValueError(
                    "Can't combine TokenSpans over different sets " "of tokens"
                )
            if (
                self.begin_token == TokenSpan.NULL_OFFSET_VALUE
                or other.begin_token == TokenSpan.NULL_OFFSET_VALUE
            ):
                return TokenSpan.make_null(self.tokens)
            else:
                return TokenSpan(
                    self.tokens,
                    min(self.begin_token, other.begin_token),
                    max(self.end_token, other.end_token),
                )
        else:
            raise NotImplementedError(
                f"Adding TokenSpan and {type(other)} " f"not yet implemented"
            )

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
class TokenSpanType(CharSpanType):
    """
    Pandas datatype for a span that represents a range of tokens within a
    target string.
    """

    @property
    def type(self):
        # The type for a single row of a column of type CharSpan
        return TokenSpan

    @property
    def name(self) -> str:
        """:return: A string representation of the dtype."""
        return "CharSpan"

    @classmethod
    def construct_array_type(cls):
        """
        See docstring in `ExtensionDType` class in `pandas/core/dtypes/base.py`
        for information about this method.
        """
        return TokenSpanArray


class TokenSpanArray(CharSpanArray):
    """
    A Pandas `ExtensionArray` that represents a column of token-based spans
    over a single target text.

    Spans are represented as `[begin_token, end_token)` intervals, where
    `begin_token` and `end_token` are token offsets into the target text.

    Null values are encoded with begin and end offsets of
    `TokenSpan.NULL_OFFSET_VALUE`.

    Fields:
    * `self._tokens`: Reference to the target string's tokens as a
        `CharSpanArray`. For now, references to different `CharSpanArray`
        objects are treated as different even if the arrays have the same
        contents.
    * `self._begin_tokens`: Numpy array of integer offsets in tokens. An offset
       of TokenSpan.NULL_OFFSET_VALUE here indicates a null value.
    * `self._end_tokens`: Numpy array of end offsets (1 + last token in span).
    """

    @staticmethod
    def from_char_offsets(tokens: CharSpanArray) -> "TokenSpanArray":
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
        tokens: CharSpanArray,
        begin_tokens: Union[np.ndarray, List[int]] = None,
        end_tokens: Union[np.ndarray, List[int]] = None,
    ):
        """
        :param tokens: Character-level span information about the underlying
        tokens.

        :param begin_tokens: Array of begin offsets measured in tokens
        :param end_tokens: Array of end offsets measured in tokens
        """
        begin_tokens = (
            np.array(begin_tokens) if isinstance(begin_tokens, list) else begin_tokens
        )
        end_tokens = (
            np.array(end_tokens) if isinstance(end_tokens, list) else end_tokens
        )
        self._tokens = tokens  # Type: CharSpanArray
        self._begin_tokens = begin_tokens  # Type: np.ndarray
        self._end_tokens = end_tokens  # Type: np.ndarray
        # Cached hash value
        self._hash = None

    # Overrides of superclass methods go here.

    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        return TokenSpanType()

    def __len__(self) -> int:
        return len(self._begin_tokens)

    def __getitem__(self, item) -> Union[TokenSpan, "TokenSpanArray"]:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(item, int):
            return TokenSpan(
                self._tokens, int(self._begin_tokens[item]), int(self._end_tokens[item])
            )
        else:
            # item not an int --> assume it's a numpy-compatible index
            return TokenSpanArray(
                self._tokens, self.begin_token[item], self.end_token[item]
            )

    def __setitem__(self, key: Union[int, np.ndarray, list, slice], value: Any) -> None:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        expected_key_types = (int, np.ndarray, list, slice)
        if not isinstance(key, expected_key_types):
            raise NotImplementedError(
                f"Don't understand key type "
                f"'{type(key)}'; should be one of "
                f"{expected_key_types}"
            )
        if value is None:
            self._begin_tokens[key] = TokenSpan.NULL_OFFSET_VALUE
            self._end_tokens[key] = TokenSpan.NULL_OFFSET_VALUE
        elif not isinstance(value, TokenSpan):
            raise ValueError(
                f"Attempted to set element of TokenSpanArray with"
                f"an object of type {type(value)}"
            )
        else:
            self._begin_tokens[key] = value.begin_token
            self._end_tokens[key] = value.end_token
        self._clear_cached_properties()

    def __eq__(self, other):
        """
        Pandas/Numpy-style array/series comparison function.

        :param other: Second operand of a Pandas "==" comparison with the series
        that wraps this TokenSpanArray.

        :return: Returns a boolean mask indicating which rows match `other`.
        """
        if isinstance(other, TokenSpan):
            if not self.tokens.equals(other.tokens):
                return np.zeros(self._begin_tokens.shape, dtype=np.bool)
            mask = np.full(len(self), True, dtype=np.bool)
            mask[self.begin_token != other.begin_token] = False
            mask[self.end_token != other.end_token] = False
            return mask
        elif isinstance(other, TokenSpanArray):
            if len(self) != len(other):
                raise ValueError(
                    "Can't compare arrays of differing lengths "
                    "{} and {}".format(len(self), len(other))
                )
            if not self.tokens.equals(other.tokens):
                return np.zeros(self._begin_tokens.shape, dtype=np.bool)
            return np.logical_and(
                self.begin_token == self.begin_token, self.end_token == self.end_token
            )
        else:
            # TODO: Return False here once we're sure that this
            #  function is catching all the comparisons that really matter.
            raise ValueError(
                "Don't know how to compare objects of type "
                "'{}' and '{}'".format(type(self), type(other))
            )

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (self._tokens, self._begin_tokens.tobytes(), self._end_tokens.tobytes())
            )
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
        # `fill_value`.  Any other negative values raise a ``ValueError``."
        if fill_value is None or np.math.isnan(fill_value):
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
        if isinstance(other, (TokenSpanArray, TokenSpan)):
            # Use token offsets when available.
            return self.end_token <= other.begin_token
        elif isinstance(other, (CharSpanArray, CharSpan)):
            return self.end <= other.begin
        else:
            raise ValueError(
                "'<' relationship not defined for {} and {} "
                "of types {} and {}"
                "".format(self, other, type(self), type(other))
            )

    def __le__(self, other):
        # TODO: Figure out what the semantics of this operation should be.
        raise NotImplementedError()

    def __ge__(self, other):
        # TODO: Figure out what the semantics of this operation should be.
        raise NotImplementedError()

    def __add__(self, other):
        """
        span1 + span2 == minimal span that covers both spans
        :param other: Other span or array of spans to add to this array's spans.
        :return: minimal span that covers both spans
        """
        if not isinstance(other, (TokenSpan, TokenSpanArray)):
            # TODO: Support adding CharSpan and TokenSpan to TokenSpanArray
            raise NotImplementedError(
                f"Adding TokenSpanArray and {type(other)} " f"not yet implemented"
            )

        if not self.tokens.equals(other.tokens):
            raise ValueError(
                f"Cannot add TokenSpans over different sets of tokens "
                f"(got {self.tokens} and {other.tokens})"
            )
        new_begin_tokens = np.minimum(self.begin_token, other.begin_token)
        new_end_tokens = np.maximum(self.end_token, other.end_token)
        return TokenSpanArray(self.tokens, new_begin_tokens, new_end_tokens)

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

    @property
    def tokens(self) -> CharSpanArray:
        return self._tokens

    @property
    def target_text(self) -> str:
        """
        :return: the common "document" text that the spans in this array
        reference.
        """
        return self._tokens.target_text

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
        result = self._tokens.begin[self.begin_token]
        # Correct for null values
        result[self.nulls_mask] = TokenSpan.NULL_OFFSET_VALUE
        return result

    @memoized_property
    def end(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span ends.
        """
        # Start out with the end of the last token in each span.
        result = self._tokens.end[self.end_token - 1]
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

    def _repr_html_(self) -> str:
        """
        HTML pretty-printing of a series of spans for Jupyter notebooks.
        """
        return util.pretty_print_html(self)

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
