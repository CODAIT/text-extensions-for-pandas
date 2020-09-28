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

from text_extensions_for_pandas.spanner.extract import extract_regex_tok
from text_extensions_for_pandas.io.spacy import make_tokens
from text_extensions_for_pandas.util import TestBase
from text_extensions_for_pandas.array.token_span import (TokenSpan, TokenSpanArray)
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
_TOKENS_ARRAY = _TOKENS_SERIES.values  # Type: SpanArray
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



    def test_overlaps_join(self):
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

        result1 = overlap_join(join_arg, _CAPS_WORD["match"])
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

        result2 = overlap_join(_CAPS_WORD["match"], join_arg)
        self.assertEqual(
            str(result2),
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
