#
# token_span.py
#
# Part of pandas_text
#
# Pandas extensions to support columns of spans with token offsets.
#

import pandas as pd
import numpy as np
from memoized_property import memoized_property

# Internal imports
import pandas_text.util as util
from pandas_text.char_span import CharSpan, CharSpanArray, CharSpanType


class TokenSpan(CharSpan):
    """
    Python object representation of a single span with token offsets; that
    is, a single row of a `TokenSpanArray`.

    This class is also a subclass of `CharSpan` and can return character-level
    information.
    """

    def __init__(self, tokens: CharSpanArray, begin_token: int, end_token: int):
        """
        :param tokens: Tokenization information about the document, including
        the target text.
        :param begin_token: Begin offset (inclusive) within the tokenized text
        :param end_token: End offset; exclusive, one past the last token
        """
        begin_char_off = tokens.begin[begin_token]
        end_char_off = (begin_char_off if begin_token == end_token
                        else tokens.end[end_token - 1])
        super().__init__(tokens.target_text, begin_char_off, end_char_off)
        self._tokens = tokens
        self._begin_token = begin_token
        self._end_token = end_token

    def __repr__(self) -> str:
        return "[{}, {}): '{}'".format(self.begin, self.end, self.covered_text)

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


class TokenSpanArray(pd.api.extensions.ExtensionArray):
    """
    A Pandas `ExtensionArray` that represents a column of token-based spans
    over a single target text.

    Spans are represented as `[begin_token, end_token)` intervals, where
    `begin_token` and `end_token` are token offsets into the target text.
    """

    def __init__(self, tokens: CharSpanArray,
                 begin_tokens: np.ndarray, end_tokens: np.ndarray):
        self._tokens = tokens
        self._begin_tokens = begin_tokens
        self._end_tokens = end_tokens

    ##############################
    # Mandatory fields/methods
    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        return TokenSpanType()

    def __len__(self) -> int:
        return len(self._begin_tokens)

    def __getitem__(self, item) -> TokenSpan:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(item, int):
            return TokenSpan(self._tokens, int(self._begin_tokens[item]),
                             int(self._end_tokens[item]))
        else:
            raise ValueError(
                "Indexing by item type '{}' not supported".format(type(item)))

    #########################################
    # Special fields/methods for span columns

    @property
    def target_text(self) -> str:
        """
        :return: the common "document" text that the spans in this array
        reference.
        """
        return self._tokens.target_text

    @memoized_property
    def begin(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span begins.
        """
        return self._tokens.begin[self.begin_token]

    @memoized_property
    def end(self) -> np.ndarray:
        """
        :return: the *character* offsets of the span ends.
        """
        # Start out with the end of the last token in each span.
        ret = self._tokens.end[self.end_token]
        # Replace end offset with begin offset wherever the length in tokens
        # is zero.
        mask = (self.end_token == self.begin_token)
        ret[mask] = self.begin[mask]
        return ret

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
            (self.begin.reshape((-1, 1)), self.end.reshape((-1, 1))),
            axis=1)

    @property
    def covered_text(self) -> np.ndarray:
        """
        Returns an array of the substrings of `target_text` corresponding to
        the spans in this array.
        """
        # TODO: Vectorized version of this
        text = self.target_text
        return np.array([
            text[s[0]:s[1]] for s in self.as_tuples()
        ])

    def as_frame(self) -> pd.DataFrame:
        """
        Returns a dataframe representation of this column based on Python
        atomic types.
        """
        return pd.DataFrame({
            "begin": self.begin,
            "end": self.end,
            "begin_token": self.begin_token,
            "end_token": self.end_token,
            "covered_text": self.covered_text
        })

    def _repr_html_(self) -> str:
        """
        HTML pretty-printing of a series of spans for Jupyter notebooks.
        """
        return util.pretty_print_html(self)
