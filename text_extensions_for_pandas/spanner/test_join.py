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

import pandas as pd
import regex
from spacy.lang.en import English

from text_extensions_for_pandas import SpanArray, SpanDtype
from text_extensions_for_pandas.spanner.extract import extract_regex_tok
from text_extensions_for_pandas.io.spacy import make_tokens
from text_extensions_for_pandas.util import TestBase
from text_extensions_for_pandas.array.token_span import (TokenSpan, TokenSpanArray,
                                                         TokenSpanDtype)
from text_extensions_for_pandas.spanner.join import (
    adjacent_join,
    contain_join,
    overlap_join,
)

# SpaCy tokenizer (only) setup
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
_tokenizer = nlp.Defaults.create_tokenizer(nlp)

# Build up some example relations for the tests in this file
_TEXT = """
In AD 932, King Arthur and his squire, Patsy, travel throughout Britain 
searching for men to join the Knights of the Round Table. Along the way, he 
recruits Sir Bedevere the Wise, Sir Lancelot the Brave, Sir Galahad the Pure...
"""
_TOKENS_SERIES = make_tokens(_TEXT, _tokenizer)
_TOKENS_ARRAY = _TOKENS_SERIES.array  # type: SpanArray
_TOKEN_SPANS_ARRAY = TokenSpanArray.from_char_offsets(_TOKENS_ARRAY)
_CAPS_WORD = extract_regex_tok(_TOKENS_ARRAY, regex.compile("[A-Z][a-z]*"))
_CAPS_WORDS = extract_regex_tok(
    _TOKENS_ARRAY, regex.compile("[A-Z][a-z]*(\\s([A-Z][a-z]*))*"), 1, 2
)
_THE = extract_regex_tok(_TOKENS_ARRAY, regex.compile("[Tt]he"))


class JoinTest(TestBase):
    def setUp(self):
        # Make it easier to see what's going on with join results
        self._prev_token_offsets_flag_value = TokenSpan.USE_TOKEN_OFFSETS_IN_REPR
        TokenSpan.USE_TOKEN_OFFSETS_IN_REPR = True

    def tearDown(self):
        # Restore TokenSpan repr formatting to avoid messing up other tests.
        TokenSpan.USE_TOKEN_OFFSETS_IN_REPR = self._prev_token_offsets_flag_value

    def test_adjacent_join(self):

        result1 = adjacent_join(_THE["match"], _CAPS_WORD["match"])
        self.assertEqual(
            str(result1),
            textwrap.dedent(
                """\
                         first               second
            0  [22, 23): 'the'  [23, 24): 'Knights'
            1  [25, 26): 'the'    [26, 27): 'Round'
            2  [38, 39): 'the'     [39, 40): 'Wise'
            3  [43, 44): 'the'    [44, 45): 'Brave'
            4  [48, 49): 'the'     [49, 50): 'Pure'"""
            ),
        )
        result2 = adjacent_join(_THE["match"], _CAPS_WORD["match"], max_gap=3)
        self.assertEqual(
            str(result2),
            textwrap.dedent(
                """\
                          first                second
            0   [22, 23): 'the'   [23, 24): 'Knights'
            1   [22, 23): 'the'     [26, 27): 'Round'
            2   [25, 26): 'the'     [26, 27): 'Round'
            3   [25, 26): 'the'     [27, 28): 'Table'
            4   [25, 26): 'the'     [29, 30): 'Along'
            5   [38, 39): 'the'      [39, 40): 'Wise'
            6   [38, 39): 'the'       [41, 42): 'Sir'
            7   [38, 39): 'the'  [42, 43): 'Lancelot'
            8   [43, 44): 'the'     [44, 45): 'Brave'
            9   [43, 44): 'the'       [46, 47): 'Sir'
            10  [43, 44): 'the'   [47, 48): 'Galahad'
            11  [48, 49): 'the'      [49, 50): 'Pure'"""
            ),
        )
        result3 = adjacent_join(
            _THE["match"], _CAPS_WORD["match"], min_gap=3, max_gap=6
        )
        self.assertEqual(
            str(result3),
            textwrap.dedent(
                """\
                         first                second
            0  [22, 23): 'the'     [26, 27): 'Round'
            1  [22, 23): 'the'     [27, 28): 'Table'
            2  [22, 23): 'the'     [29, 30): 'Along'
            3  [25, 26): 'the'     [29, 30): 'Along'
            4  [30, 31): 'the'       [36, 37): 'Sir'
            5  [30, 31): 'the'  [37, 38): 'Bedevere'
            6  [38, 39): 'the'  [42, 43): 'Lancelot'
            7  [38, 39): 'the'     [44, 45): 'Brave'
            8  [43, 44): 'the'   [47, 48): 'Galahad'
            9  [43, 44): 'the'      [49, 50): 'Pure'"""
            ),
        )

    def _make_join_arg(self) -> pd.Series:
        """
        Shared example join argument used by most of the test cases that follow.
        """
        return pd.Series(
            TokenSpanArray._from_sequence(
                [
                    TokenSpan(_TOKENS_ARRAY, 23, 28),  # Knights of the Round Table
                    TokenSpan(_TOKENS_ARRAY, 17, 19),  # searching for
                    TokenSpan(_TOKENS_ARRAY, 1, 2),  # In
                    TokenSpan(_TOKENS_ARRAY, 1, 2),  # In (second copy)
                    TokenSpan(_TOKENS_ARRAY, 42, 45),  # Lancelot the Brave
                ]
            )
        )

    def _make_join_arg_span_array(self) -> pd.Series:
        """
        Version of the result of _make_join_arg() as a SpanArray instead of a
        TokenSpanArray.
        """
        # noinspection PyTypeChecker
        token_span_array = self._make_join_arg().array  # type: TokenSpanArray
        span_array = SpanArray(_TEXT, token_span_array.begin, token_span_array.end)
        return pd.Series(span_array)

    @staticmethod
    def _make_empty_series() -> pd.Series:
        """
        Zero-length TokenSpanArray wrapped in a series. Note that this array has
        zero spans but *does* contain token and text information.
        """
        return pd.Series(
            TokenSpanArray(_TOKENS_ARRAY, [], [])
        )

    @staticmethod
    def _make_empty_series_span_array() -> pd.Series:
        return pd.Series(SpanArray(_TEXT, [], []))

    def test_overlaps_join_left_spans_longer(self):
        result = overlap_join(self._make_join_arg(), _CAPS_WORD["match"])
        self.assertEqual(
            str(result),
            # Note that offsets are in *tokens*, because this test class enables the
            # TokenSpan.USE_TOKEN_OFFSETS_IN_REPR flag.
            textwrap.dedent(
                """\
                                                first                second
            0  [23, 28): 'Knights of the Round Table'   [23, 24): 'Knights'
            1  [23, 28): 'Knights of the Round Table'     [26, 27): 'Round'
            2  [23, 28): 'Knights of the Round Table'     [27, 28): 'Table'
            3                            [1, 2): 'In'          [1, 2): 'In'
            4                            [1, 2): 'In'          [1, 2): 'In'
            5          [42, 45): 'Lancelot the Brave'  [42, 43): 'Lancelot'
            6          [42, 45): 'Lancelot the Brave'     [44, 45): 'Brave'"""
            ),
        )

    def test_overlaps_join_right_spans_longer(self):
        result = overlap_join(_CAPS_WORD["match"], self._make_join_arg())
        self.assertEqual(
            str(result),
            textwrap.dedent(
                """\
                              first                                  second
            0          [1, 2): 'In'                            [1, 2): 'In'
            1          [1, 2): 'In'                            [1, 2): 'In'
            2   [23, 24): 'Knights'  [23, 28): 'Knights of the Round Table'
            3     [26, 27): 'Round'  [23, 28): 'Knights of the Round Table'
            4     [27, 28): 'Table'  [23, 28): 'Knights of the Round Table'
            5  [42, 43): 'Lancelot'          [42, 45): 'Lancelot the Brave'
            6     [44, 45): 'Brave'          [42, 45): 'Lancelot the Brave'"""
            ),
        )

    def test_overlaps_join_left_char_spans(self):
        result = overlap_join(self._make_join_arg_span_array(), _CAPS_WORD["match"])
        self.assertEqual(
            str(result),
            # Note the span offsets instead of token offsets in "first"
            textwrap.dedent(
                """\
                                                      first                second
                0  [104, 130): 'Knights of the Round Table'   [23, 24): 'Knights'
                1  [104, 130): 'Knights of the Round Table'     [26, 27): 'Round'
                2  [104, 130): 'Knights of the Round Table'     [27, 28): 'Table'
                3                              [1, 3): 'In'          [1, 2): 'In'
                4                              [1, 3): 'In'          [1, 2): 'In'
                5          [187, 205): 'Lancelot the Brave'  [42, 43): 'Lancelot'
                6          [187, 205): 'Lancelot the Brave'     [44, 45): 'Brave'"""
            ),
        )

    def test_overlaps_join_empty_input(self):
        # Empty input should produce empty result with valid token info.
        result1 = overlap_join(self._make_empty_series(), self._make_join_arg())
        self.assertEqual(len(result1.index), 0)
        self.assertIsInstance(result1["first"].dtype, TokenSpanDtype)
        self.assertIsInstance(result1["second"].dtype, TokenSpanDtype)

        result2 = overlap_join(self._make_join_arg(), self._make_empty_series())
        self.assertEqual(len(result2.index), 0)
        self.assertIsInstance(result2["first"].dtype, TokenSpanDtype)
        self.assertIsInstance(result2["second"].dtype, TokenSpanDtype)

        result3 = overlap_join(self._make_empty_series(), self._make_empty_series())
        self.assertEqual(len(result3.index), 0)
        self.assertIsInstance(result3["first"].dtype, TokenSpanDtype)
        self.assertIsInstance(result3["second"].dtype, TokenSpanDtype)

    def test_overlaps_join_empty_input_span_array(self):
        # The previous test, but with character offsets instead of token offsets
        result1 = overlap_join(self._make_empty_series_span_array(),
                               self._make_join_arg())
        self.assertEqual(len(result1.index), 0)
        self.assertIsInstance(result1["first"].dtype, SpanDtype)
        self.assertIsInstance(result1["second"].dtype, TokenSpanDtype)

        result2 = overlap_join(self._make_join_arg(),
                               self._make_empty_series_span_array())
        self.assertEqual(len(result2.index), 0)
        self.assertIsInstance(result2["first"].dtype, TokenSpanDtype)
        self.assertIsInstance(result2["second"].dtype, SpanDtype)

        result3 = overlap_join(self._make_empty_series_span_array(),
                               self._make_empty_series_span_array())
        self.assertEqual(len(result3.index), 0)
        self.assertIsInstance(result3["first"].dtype, SpanDtype)
        self.assertIsInstance(result3["second"].dtype, SpanDtype)

    def test_contain_join(self):
        join_arg = pd.Series(
            TokenSpanArray._from_sequence(
                [
                    TokenSpan(_TOKENS_ARRAY, 23, 28),  # Knights of the Round Table
                    TokenSpan(_TOKENS_ARRAY, 17, 19),  # searching for
                    TokenSpan(_TOKENS_ARRAY, 1, 2),  # In
                    TokenSpan(_TOKENS_ARRAY, 1, 2),  # In (second copy)
                    TokenSpan(_TOKENS_ARRAY, 42, 45),  # Lancelot the Brave
                ]
            )
        )

        result1 = contain_join(join_arg, _CAPS_WORD["match"])
        self.assertEqual(
            str(result1),
            textwrap.dedent(
                """\
                                                    first                second
                0  [23, 28): 'Knights of the Round Table'   [23, 24): 'Knights'
                1  [23, 28): 'Knights of the Round Table'     [26, 27): 'Round'
                2  [23, 28): 'Knights of the Round Table'     [27, 28): 'Table'
                3                            [1, 2): 'In'          [1, 2): 'In'
                4                            [1, 2): 'In'          [1, 2): 'In'
                5          [42, 45): 'Lancelot the Brave'  [42, 43): 'Lancelot'
                6          [42, 45): 'Lancelot the Brave'     [44, 45): 'Brave'"""
            ),
        )

        result2 = contain_join(_CAPS_WORD["match"], join_arg)
        self.assertEqual(
            str(result2),
            textwrap.dedent(
                """\
                          first        second
                0  [1, 2): 'In'  [1, 2): 'In'
                1  [1, 2): 'In'  [1, 2): 'In'"""
            ),
        )


if __name__ == "__main__":
    unittest.main()
