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
# accessor.py
#
# Part of text_extensions_for_pandas
#
# Pandas accessors to create namespaces for extension types' type-specific
# functionality.
#

import pandas as pd

# Internal imports
from text_extensions_for_pandas.array.span import SpanDtype


@pd.api.extensions.register_series_accessor("span")
class SpanAccessor:
    """
    Pandas custom series accessor that creates a "span" namespace that can be
    used to call span-specific methods of the `SpanArray`
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
        if not isinstance(dtype, SpanDtype):
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
