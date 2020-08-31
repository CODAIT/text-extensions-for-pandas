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

import textwrap
import unittest

from text_extensions_for_pandas.util import TestBase
from text_extensions_for_pandas.spanner.consolidate import *
from text_extensions_for_pandas.array.span import Span, SpanArray

import pandas as pd


class ConsolidateTest(TestBase):
    def test_left_to_right(self):
        test_text = "Is it weird in here, or is it just me?"
        spans = [
            Span(test_text, 0, 3),
            Span(test_text, 2, 3),
            Span(test_text, 3, 3),
            Span(test_text, 1, 3),
            Span(test_text, 0, 4),  # index 4
            Span(test_text, 5, 7),  # index 5
            Span(test_text, 6, 9),
            Span(test_text, 8, 9),  # index 7
        ]
        df = pd.DataFrame({
            "s": SpanArray._from_sequence(spans),
            "ix": range(len(spans))
        })
        c_df = consolidate(df, on="s", how="left_to_right")
        self._assertArrayEquals(list(c_df.index), [4, 5, 7])
