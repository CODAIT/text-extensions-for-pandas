#
# char_span.py
#
# Part of pandas_text
#
# Pandas extensions to support columns of spans with character offsets.
#

import pandas as pd
import numpy as np
from memoized_property import memoized_property
from typing import *

# Internal imports
import pandas_text.util as util


class CharSpan:
    """
    Python object representation of a single span with character offsets; that
    is, a single row of a `CharSpanArray`
    """

    def __init__(self, text: str, begin: int, end: int):
        """
        Args:
            text: target document text on which the span is defined
            begin: Begin offset (inclusive) within `text`
            end: End offset (exclusive, one past the last char) within `text`
        """
        self._text = text
        self._begin = begin
        self._end = end

    def __repr__(self) -> str:
        return "[{}, {}): '{}'".format(self.begin, self.end, self.covered_text)

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
        Returns the substring of `self.target_text` that this `CharSpan`
        represents.
        """
        return self.target_text[self.begin:self.end]


@pd.api.extensions.register_extension_dtype
class CharSpanType(pd.api.extensions.ExtensionDtype):
    """
    Panda datatype for a span that represents a range of characters within a
    target string.
    """

    @property
    def type(self):
        # The type for a single row of a column of type CharSpan
        return CharSpan

    @property
    def name(self) -> str:
        """A string representation of the dtype."""
        return "CharSpan"


class CharSpanArray(pd.api.extensions.ExtensionArray):
    """
    A Pandas `ExtensionArray` that represents a column of character-based spans
    over a single target text.

    Spans are represented as `[begin, end)` intervals, where `begin` and `end`
    are character offsets into the target text.
    """

    def __init__(self, text: str, begins: np.ndarray, ends: np.ndarray):
        self._text = text
        self._begins = begins
        self._ends = ends

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence[pd.api.extensions.ExtensionArray]
    ) -> pd.api.extensions.ExtensionArray:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        text = {a.target_text for a in to_concat}
        if len(text) != 1:
            raise ValueError("CharSpans must all be over the same target text")
        text = text.pop()

        begins = np.concatenate([a.begin for a in to_concat])
        ends = np.concatenate([a.end for a in to_concat])
        return CharSpanArray(text, begins, ends)

    def isna(self) -> np.array:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # No na's allowed at the moment.
        return np.repeat(False, len(self))

    def copy(self) -> "CharSpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        ret = CharSpanArray(
            self.target_text,
            self.begin.copy(),
            self.end.copy()
        )
        # TODO: Copy cached properties too
        return ret

    def take(
        self, indices: Sequence[int], allow_fill: bool = False,
        fill_value: Any = None
    ) -> "CharSpanArray":
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        # TODO: When TokenSpanArray.take() implements allow_fill properly, copy
        #  that implementation here.
        if allow_fill:
            if np.all(np.array(indices) >= 0):
                return CharSpanArray(self.target_text,
                                     np.take(self.begin, indices),
                                     np.take(self.end, indices))
            else:
                raise ValueError("allow_fill mode not implemented "
                                 "(indices {})".format(indices))
        else:
            return CharSpanArray(self.target_text,
                                 np.take(self.begin, indices),
                                 np.take(self.end, indices))


    @property
    def dtype(self) -> pd.api.extensions.ExtensionDtype:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        return CharSpanType()

    def __len__(self) -> int:
        return len(self._begins)

    def __getitem__(self, item) -> CharSpan:
        """
        See docstring in `ExtensionArray` class in `pandas/core/arrays/base.py`
        for information about this method.
        """
        if isinstance(item, int):
            return CharSpan(self._text, int(self._begins[item]),
                            int(self._ends[item]))
        else:
            # item not an int --> assume it's a numpy-compatible index
            return CharSpanArray(self.target_text,
                                 self.begin[item], self.end[item])

    @property
    def target_text(self) -> str:
        """
        Returns the common "document" text that the spans in this array
        reference.
        """
        return self._text

    @property
    def begin(self) -> np.ndarray:
        return self._begins

    @property
    def end(self) -> np.ndarray:
        return self._ends

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
        :return: an array of the substrings of `target_text` corresponding to
        the spans in this array.
        """
        # TODO: Vectorized version of this
        text = self.target_text
        return np.array([
            text[s[0]:s[1]] for s in self.as_tuples()
        ])

    @memoized_property
    def normalized_covered_text(self) -> np.ndarray:
        """
        :return: A normalized version of the covered text of the spans in this
          array. Currently "normalized" means "lowercase".
        """
        return np.char.lower(self.covered_text)

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

    def _repr_html_(self) -> str:
        """
        HTML pretty-printing of a series of spans for Jupyter notebooks.
        """
        return util.pretty_print_html(self)
