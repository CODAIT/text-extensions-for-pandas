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

import numpy as np
import unittest
import textwrap

from text_extensions_for_pandas.io.conll import *
from text_extensions_for_pandas.io.spacy import make_tokens_and_features

import spacy

_SPACY_LANGUAGE_MODEL = spacy.load("en_core_web_sm")


class IOTest(unittest.TestCase):

    def test_iob_to_spans(self):
        df = make_tokens_and_features(
            textwrap.dedent(
                """\
            The Bermuda Triangle got tired of warm weather. 
            It moved to Alaska. Now Santa Claus is missing.
            -- Steven Wright"""
            ),
            _SPACY_LANGUAGE_MODEL,
        )
        spans = iob_to_spans(df)
        # print(f"****{spans}****")
        self.assertEqual(
            str(spans),
            textwrap.dedent(
                """\
                                    token_span ent_type
                0           [61, 67): 'Alaska'      GPE
                1      [73, 84): 'Santa Claus'      GPE
                2  [100, 113): 'Steven Wright'   PERSON"""
            ),
        )

    def test_conll_2003_to_dataframes(self):
        dfs = conll_2003_to_dataframes("test_data/io/test_conll/conll03_test.txt")
        self.assertEqual(len(dfs), 2)
        self.assertEqual(
            dfs[0]["char_span"].values.target_text,
            textwrap.dedent(
                """\
                Who is General Failure and why is he reading my hard disk ?
                If Barbie is so popular , why do you have to buy her friends ?"""
            )
        )
        self.assertEqual(
            dfs[1]["char_span"].values.target_text,
            "I'd kill for a Nobel Peace Prize ."
        )
        # print(f"***{repr(dfs[0])}***")
        self.assertEqual(
            repr(dfs[0]),
            textwrap.dedent(
                """\
                                char_span             token_span ent_iob ent_type
                0           [0, 3): 'Who'          [0, 3): 'Who'       O     None
                1            [4, 6): 'is'           [4, 6): 'is'       O     None
                2      [7, 14): 'General'     [7, 14): 'General'       B      PER
                3     [15, 22): 'Failure'    [15, 22): 'Failure'       I      PER
                4         [23, 26): 'and'        [23, 26): 'and'       O     None
                5         [27, 30): 'why'        [27, 30): 'why'       O     None
                6          [31, 33): 'is'         [31, 33): 'is'       B      FOO
                7          [34, 36): 'he'         [34, 36): 'he'       B      BAR
                8     [37, 44): 'reading'    [37, 44): 'reading'       O     None
                9          [45, 47): 'my'         [45, 47): 'my'       O     None
                10       [48, 52): 'hard'       [48, 52): 'hard'       B      FAB
                11       [53, 57): 'disk'       [53, 57): 'disk'       B      FAB
                12          [58, 59): '?'          [58, 59): '?'       O     None
                13         [60, 62): 'If'         [60, 62): 'If'       O     None
                14     [63, 69): 'Barbie'     [63, 69): 'Barbie'       B      PER
                15         [70, 72): 'is'         [70, 72): 'is'       O     None
                16         [73, 75): 'so'         [73, 75): 'so'       O     None
                17    [76, 83): 'popular'    [76, 83): 'popular'       O     None
                18          [84, 85): ','          [84, 85): ','       O     None
                19        [86, 89): 'why'        [86, 89): 'why'       O     None
                20         [90, 92): 'do'         [90, 92): 'do'       O     None
                21        [93, 96): 'you'        [93, 96): 'you'       O     None
                22      [97, 101): 'have'      [97, 101): 'have'       O     None
                23       [102, 104): 'to'       [102, 104): 'to'       O     None
                24      [105, 108): 'buy'      [105, 108): 'buy'       O     None
                25      [109, 112): 'her'      [109, 112): 'her'       O     None
                26  [113, 120): 'friends'  [113, 120): 'friends'       O     None
                27        [121, 122): '?'        [121, 122): '?'       O     None"""
            )
        )
        # print(f"***{repr(dfs[1])}***")
        self.assertEqual(
            repr(dfs[1]),
            textwrap.dedent(
                """\
                           char_span         token_span ent_iob ent_type
                0      [0, 3): 'I'd'      [0, 3): 'I'd'       O     None
                1     [4, 8): 'kill'     [4, 8): 'kill'       O     None
                2     [9, 12): 'for'     [9, 12): 'for'       O     None
                3      [13, 14): 'a'      [13, 14): 'a'       O     None
                4  [15, 20): 'Nobel'  [15, 20): 'Nobel'       B     MISC
                5  [21, 26): 'Peace'  [21, 26): 'Peace'       I     MISC
                6  [27, 32): 'Prize'  [27, 32): 'Prize'       I     MISC
                7      [33, 34): '.'      [33, 34): '.'       O     None"""
            )
        )




if __name__ == "__main__":
    unittest.main()
