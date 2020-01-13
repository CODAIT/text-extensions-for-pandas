#
# span.py
#
# Part of pandas_text
#
# Functionality common to both character- and token-based span types.
#

import pandas as pd

# Internal imports
from pandas_text.char_span import CharSpanType


@pd.api.extensions.register_series_accessor("span")
class SpanAccessor:
    """
    Pandas custom series accessor that creates a "span" namespace that can be
    used to call span-specific methods of the `CharSpanArray`
    or `TokenSpanArray` that backs a particular column of spans.
    """

    def __init__(self, obj):
        """
        Args:
            obj: Pandas will pass a pointer to the `pd.Series` object under this
                 argument
        """
        self._validate(obj)
        self._values = obj.values
        self._index = obj.index
        self._name = obj.name

    @staticmethod
    def _validate(obj):
        dtype = getattr(obj, "dtype", obj)
        if not isinstance(dtype, CharSpanType):
            raise AttributeError("Cannot use 'span' accessor on objects of "
                                 "dtype '{}'.".format(obj.dtype))

    @property
    def begin(self):
        return pd.Series(self._values.begin)

    @property
    def end(self):
        return pd.Series(self._values.end)

    @property
    def covered_text(self):
        return pd.Series(self._values.covered_text)
