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

    def setUp(self):
        # Ensure that diffs are consistent
        pd.set_option("display.max_columns", 250)

    def tearDown(self):
        pd.reset_option("display.max_columns")

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

    def test_spans_to_iob(self):
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
        self.assertTrue('ent_iob' in df.columns)
        self.assertTrue('token_span' in spans.columns)
        df_result = spans_to_iob(spans)
        self.assertTrue('ent_iob' in df_result.columns)
        pd.testing.assert_series_equal(df['ent_iob'], df_result['ent_iob'])

    def test_conll_2003_to_dataframes(self):
        dfs = conll_2003_to_dataframes("test_data/io/test_conll/conll03_test.txt")
        self.assertEqual(len(dfs), 2)
        self.assertEqual(
            dfs[0]["char_span"].values.target_text,
            textwrap.dedent(
                """\
                Who is General Failure and why is he reading my hard disk?
                If Barbie is so popular, why do you have to buy her friends?"""
            )
        )
        self.assertEqual(
            dfs[1]["char_span"].values.target_text,
            "-DOCSTART-\nI'd kill for a Nobel Peace Prize."
        )
        # print(f"***{repr(dfs[0])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(dfs[0]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                char_span             token_span ent_iob ent_type  \\
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
                12          [57, 58): '?'          [57, 58): '?'       O     None   
                13         [59, 61): 'If'         [59, 61): 'If'       O     None   
                14     [62, 68): 'Barbie'     [62, 68): 'Barbie'       B      PER   
                15         [69, 71): 'is'         [69, 71): 'is'       O     None   
                16         [72, 74): 'so'         [72, 74): 'so'       O     None   
                17    [75, 82): 'popular'    [75, 82): 'popular'       O     None   
                18          [82, 83): ','          [82, 83): ','       O     None   
                19        [84, 87): 'why'        [84, 87): 'why'       O     None   
                20         [88, 90): 'do'         [88, 90): 'do'       O     None   
                21        [91, 94): 'you'        [91, 94): 'you'       O     None   
                22       [95, 99): 'have'       [95, 99): 'have'       O     None   
                23       [100, 102): 'to'       [100, 102): 'to'       O     None   
                24      [103, 106): 'buy'      [103, 106): 'buy'       O     None   
                25      [107, 110): 'her'      [107, 110): 'her'       O     None   
                26  [111, 118): 'friends'  [111, 118): 'friends'       O     None   
                27        [118, 119): '?'        [118, 119): '?'       O     None   
                
                                                             sentence  
                0   [0, 58): 'Who is General Failure and why is he...  
                1   [0, 58): 'Who is General Failure and why is he...  
                2   [0, 58): 'Who is General Failure and why is he...  
                3   [0, 58): 'Who is General Failure and why is he...  
                4   [0, 58): 'Who is General Failure and why is he...  
                5   [0, 58): 'Who is General Failure and why is he...  
                6   [0, 58): 'Who is General Failure and why is he...  
                7   [0, 58): 'Who is General Failure and why is he...  
                8   [0, 58): 'Who is General Failure and why is he...  
                9   [0, 58): 'Who is General Failure and why is he...  
                10  [0, 58): 'Who is General Failure and why is he...  
                11  [0, 58): 'Who is General Failure and why is he...  
                12  [0, 58): 'Who is General Failure and why is he...  
                13  [59, 119): 'If Barbie is so popular, why do yo...  
                14  [59, 119): 'If Barbie is so popular, why do yo...  
                15  [59, 119): 'If Barbie is so popular, why do yo...  
                16  [59, 119): 'If Barbie is so popular, why do yo...  
                17  [59, 119): 'If Barbie is so popular, why do yo...  
                18  [59, 119): 'If Barbie is so popular, why do yo...  
                19  [59, 119): 'If Barbie is so popular, why do yo...  
                20  [59, 119): 'If Barbie is so popular, why do yo...  
                21  [59, 119): 'If Barbie is so popular, why do yo...  
                22  [59, 119): 'If Barbie is so popular, why do yo...  
                23  [59, 119): 'If Barbie is so popular, why do yo...  
                24  [59, 119): 'If Barbie is so popular, why do yo...  
                25  [59, 119): 'If Barbie is so popular, why do yo...  
                26  [59, 119): 'If Barbie is so popular, why do yo...  
                27  [59, 119): 'If Barbie is so popular, why do yo...  """
            )
        )
        # print(f"***{repr(dfs[1])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(dfs[1]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                               char_span             token_span ent_iob ent_type  \\
                0  [0, 10): '-DOCSTART-'  [0, 10): '-DOCSTART-'       O     None   
                1        [11, 14): 'I'd'        [11, 14): 'I'd'       O     None   
                2       [15, 19): 'kill'       [15, 19): 'kill'       O     None   
                3        [20, 23): 'for'        [20, 23): 'for'       O     None   
                4          [24, 25): 'a'          [24, 25): 'a'       O     None   
                5      [26, 31): 'Nobel'      [26, 31): 'Nobel'       B     MISC   
                6      [32, 37): 'Peace'      [32, 37): 'Peace'       I     MISC   
                7      [38, 43): 'Prize'      [38, 43): 'Prize'       I     MISC   
                8          [43, 44): '.'          [43, 44): '.'       O     None   
                
                                                        sentence  
                0                          [0, 10): '-DOCSTART-'  
                1  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                2  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                3  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                4  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                5  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                6  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                7  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                8  [11, 44): 'I'd kill for a Nobel Peace Prize.'  """
            )
        )

    def test_conll_2003_output_to_dataframes(self):
        doc_dfs = conll_2003_to_dataframes("test_data/io/test_conll/conll03_test.txt")
        output_dfs = conll_2003_output_to_dataframes(
            doc_dfs, "test_data/io/test_conll/conll03_output.txt")
        self.assertEqual(len(output_dfs), 2)
        self.assertEqual(
            output_dfs[0]["char_span"].values.target_text,
            textwrap.dedent(
                """\
                Who is General Failure and why is he reading my hard disk?
                If Barbie is so popular, why do you have to buy her friends?"""
            )
        )
        self.assertEqual(
            output_dfs[1]["char_span"].values.target_text,
            "-DOCSTART-\nI'd kill for a Nobel Peace Prize."
        )
        # print(f"***{repr(output_dfs[0])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(output_dfs[0]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                char_span             token_span ent_iob ent_type  \\
                0           [0, 3): 'Who'          [0, 3): 'Who'       B     BAND   
                1            [4, 6): 'is'           [4, 6): 'is'       O     None   
                2      [7, 14): 'General'     [7, 14): 'General'       B      PER   
                3     [15, 22): 'Failure'    [15, 22): 'Failure'       I      PER   
                4         [23, 26): 'and'        [23, 26): 'and'       O     None   
                5         [27, 30): 'why'        [27, 30): 'why'       B      FOO   
                6          [31, 33): 'is'         [31, 33): 'is'       I      FOO   
                7          [34, 36): 'he'         [34, 36): 'he'       B      BAR   
                8     [37, 44): 'reading'    [37, 44): 'reading'       O     None   
                9          [45, 47): 'my'         [45, 47): 'my'       O     None   
                10       [48, 52): 'hard'       [48, 52): 'hard'       B      FAB   
                11       [53, 57): 'disk'       [53, 57): 'disk'       B      FAB   
                12          [57, 58): '?'          [57, 58): '?'       O     None   
                13         [59, 61): 'If'         [59, 61): 'If'       B      PER   
                14     [62, 68): 'Barbie'     [62, 68): 'Barbie'       I      PER   
                15         [69, 71): 'is'         [69, 71): 'is'       O     None   
                16         [72, 74): 'so'         [72, 74): 'so'       O     None   
                17    [75, 82): 'popular'    [75, 82): 'popular'       O     None   
                18          [82, 83): ','          [82, 83): ','       O     None   
                19        [84, 87): 'why'        [84, 87): 'why'       O     None   
                20         [88, 90): 'do'         [88, 90): 'do'       O     None   
                21        [91, 94): 'you'        [91, 94): 'you'       O     None   
                22       [95, 99): 'have'       [95, 99): 'have'       O     None   
                23       [100, 102): 'to'       [100, 102): 'to'       O     None   
                24      [103, 106): 'buy'      [103, 106): 'buy'       O     None   
                25      [107, 110): 'her'      [107, 110): 'her'       O     None   
                26  [111, 118): 'friends'  [111, 118): 'friends'       B      ORG   
                27        [118, 119): '?'        [118, 119): '?'       O     None   
                
                                                             sentence  
                0   [0, 58): 'Who is General Failure and why is he...  
                1   [0, 58): 'Who is General Failure and why is he...  
                2   [0, 58): 'Who is General Failure and why is he...  
                3   [0, 58): 'Who is General Failure and why is he...  
                4   [0, 58): 'Who is General Failure and why is he...  
                5   [0, 58): 'Who is General Failure and why is he...  
                6   [0, 58): 'Who is General Failure and why is he...  
                7   [0, 58): 'Who is General Failure and why is he...  
                8   [0, 58): 'Who is General Failure and why is he...  
                9   [0, 58): 'Who is General Failure and why is he...  
                10  [0, 58): 'Who is General Failure and why is he...  
                11  [0, 58): 'Who is General Failure and why is he...  
                12  [0, 58): 'Who is General Failure and why is he...  
                13  [59, 119): 'If Barbie is so popular, why do yo...  
                14  [59, 119): 'If Barbie is so popular, why do yo...  
                15  [59, 119): 'If Barbie is so popular, why do yo...  
                16  [59, 119): 'If Barbie is so popular, why do yo...  
                17  [59, 119): 'If Barbie is so popular, why do yo...  
                18  [59, 119): 'If Barbie is so popular, why do yo...  
                19  [59, 119): 'If Barbie is so popular, why do yo...  
                20  [59, 119): 'If Barbie is so popular, why do yo...  
                21  [59, 119): 'If Barbie is so popular, why do yo...  
                22  [59, 119): 'If Barbie is so popular, why do yo...  
                23  [59, 119): 'If Barbie is so popular, why do yo...  
                24  [59, 119): 'If Barbie is so popular, why do yo...  
                25  [59, 119): 'If Barbie is so popular, why do yo...  
                26  [59, 119): 'If Barbie is so popular, why do yo...  
                27  [59, 119): 'If Barbie is so popular, why do yo...  """
            )
        )
        # print(f"***{repr(output_dfs[1])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(output_dfs[1]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                               char_span             token_span ent_iob ent_type  \\
                0  [0, 10): '-DOCSTART-'  [0, 10): '-DOCSTART-'       O     None   
                1        [11, 14): 'I'd'        [11, 14): 'I'd'       O     None   
                2       [15, 19): 'kill'       [15, 19): 'kill'       O     None   
                3        [20, 23): 'for'        [20, 23): 'for'       O     None   
                4          [24, 25): 'a'          [24, 25): 'a'       O     None   
                5      [26, 31): 'Nobel'      [26, 31): 'Nobel'       B     MISC   
                6      [32, 37): 'Peace'      [32, 37): 'Peace'       O     None   
                7      [38, 43): 'Prize'      [38, 43): 'Prize'       B     MISC   
                8          [43, 44): '.'          [43, 44): '.'       O     None   
                
                                                        sentence  
                0                          [0, 10): '-DOCSTART-'  
                1  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                2  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                3  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                4  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                5  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                6  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                7  [11, 44): 'I'd kill for a Nobel Peace Prize.'  
                8  [11, 44): 'I'd kill for a Nobel Peace Prize.'  """
            )
        )



if __name__ == "__main__":
    unittest.main()
