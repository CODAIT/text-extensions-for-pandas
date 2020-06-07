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


class CoNLLTest(unittest.TestCase):
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
        self.assertTrue("ent_iob" in df.columns)
        self.assertTrue("token_span" in spans.columns)
        result = spans_to_iob(spans["token_span"])
        pd.testing.assert_series_equal(df["ent_iob"], result["ent_iob"])

    def test_conll_2003_to_dataframes(self):
        dfs = conll_2003_to_dataframes("test_data/io/test_conll/conll03_test.txt")
        self.assertEqual(len(dfs), 2)
        self.assertEqual(
            dfs[0]["char_span"].values.target_text,
            textwrap.dedent(
                """\
                Who is General Failure (and why is he reading my hard disk)?
                If Barbie is so popular, why do you have to buy her friends?"""
            ),
        )
        self.assertEqual(
            dfs[1]["char_span"].values.target_text,
            "-DOCSTART-\nI'd kill for a Nobel Peace Prize.",
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
                4           [23, 24): '('          [23, 24): '('       O     None   
                5         [24, 27): 'and'        [24, 27): 'and'       O     None   
                6         [28, 31): 'why'        [28, 31): 'why'       O     None   
                7          [32, 34): 'is'         [32, 34): 'is'       B      FOO   
                8          [35, 37): 'he'         [35, 37): 'he'       B      BAR   
                9     [38, 45): 'reading'    [38, 45): 'reading'       O     None   
                10         [46, 48): 'my'         [46, 48): 'my'       O     None   
                11       [49, 53): 'hard'       [49, 53): 'hard'       B      FAB   
                12       [54, 58): 'disk'       [54, 58): 'disk'       B      FAB   
                13          [58, 59): ')'          [58, 59): ')'       O     None   
                14          [59, 60): '?'          [59, 60): '?'       O     None   
                15         [61, 63): 'If'         [61, 63): 'If'       O     None   
                16     [64, 70): 'Barbie'     [64, 70): 'Barbie'       B      PER   
                17         [71, 73): 'is'         [71, 73): 'is'       O     None   
                18         [74, 76): 'so'         [74, 76): 'so'       O     None   
                19    [77, 84): 'popular'    [77, 84): 'popular'       O     None   
                20          [84, 85): ','          [84, 85): ','       O     None   
                21        [86, 89): 'why'        [86, 89): 'why'       O     None   
                22         [90, 92): 'do'         [90, 92): 'do'       O     None   
                23        [93, 96): 'you'        [93, 96): 'you'       O     None   
                24      [97, 101): 'have'      [97, 101): 'have'       O     None   
                25       [102, 104): 'to'       [102, 104): 'to'       O     None   
                26      [105, 108): 'buy'      [105, 108): 'buy'       O     None   
                27      [109, 112): 'her'      [109, 112): 'her'       O     None   
                28  [113, 120): 'friends'  [113, 120): 'friends'       O     None   
                29        [120, 121): '?'        [120, 121): '?'       O     None   
                
                                                             sentence  
                0   [0, 60): 'Who is General Failure (and why is h...  
                1   [0, 60): 'Who is General Failure (and why is h...  
                2   [0, 60): 'Who is General Failure (and why is h...  
                3   [0, 60): 'Who is General Failure (and why is h...  
                4   [0, 60): 'Who is General Failure (and why is h...  
                5   [0, 60): 'Who is General Failure (and why is h...  
                6   [0, 60): 'Who is General Failure (and why is h...  
                7   [0, 60): 'Who is General Failure (and why is h...  
                8   [0, 60): 'Who is General Failure (and why is h...  
                9   [0, 60): 'Who is General Failure (and why is h...  
                10  [0, 60): 'Who is General Failure (and why is h...  
                11  [0, 60): 'Who is General Failure (and why is h...  
                12  [0, 60): 'Who is General Failure (and why is h...  
                13  [0, 60): 'Who is General Failure (and why is h...  
                14  [0, 60): 'Who is General Failure (and why is h...  
                15  [61, 121): 'If Barbie is so popular, why do yo...  
                16  [61, 121): 'If Barbie is so popular, why do yo...  
                17  [61, 121): 'If Barbie is so popular, why do yo...  
                18  [61, 121): 'If Barbie is so popular, why do yo...  
                19  [61, 121): 'If Barbie is so popular, why do yo...  
                20  [61, 121): 'If Barbie is so popular, why do yo...  
                21  [61, 121): 'If Barbie is so popular, why do yo...  
                22  [61, 121): 'If Barbie is so popular, why do yo...  
                23  [61, 121): 'If Barbie is so popular, why do yo...  
                24  [61, 121): 'If Barbie is so popular, why do yo...  
                25  [61, 121): 'If Barbie is so popular, why do yo...  
                26  [61, 121): 'If Barbie is so popular, why do yo...  
                27  [61, 121): 'If Barbie is so popular, why do yo...  
                28  [61, 121): 'If Barbie is so popular, why do yo...  
                29  [61, 121): 'If Barbie is so popular, why do yo...  """
            ),
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
            ),
        )

    def test_conll_2003_output_to_dataframes(self):
        doc_dfs = conll_2003_to_dataframes("test_data/io/test_conll/conll03_test.txt")
        output_dfs = conll_2003_output_to_dataframes(
            doc_dfs, "test_data/io/test_conll/conll03_output.txt"
        )
        self.assertEqual(len(output_dfs), 2)
        self.assertEqual(
            output_dfs[0]["char_span"].values.target_text,
            textwrap.dedent(
                """\
                Who is General Failure (and why is he reading my hard disk)?
                If Barbie is so popular, why do you have to buy her friends?"""
            ),
        )
        self.assertEqual(
            output_dfs[1]["char_span"].values.target_text,
            "-DOCSTART-\nI'd kill for a Nobel Peace Prize.",
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
                4           [23, 24): '('          [23, 24): '('       O     None   
                5         [24, 27): 'and'        [24, 27): 'and'       O     None   
                6         [28, 31): 'why'        [28, 31): 'why'       B      FOO   
                7          [32, 34): 'is'         [32, 34): 'is'       I      FOO   
                8          [35, 37): 'he'         [35, 37): 'he'       B      BAR   
                9     [38, 45): 'reading'    [38, 45): 'reading'       O     None   
                10         [46, 48): 'my'         [46, 48): 'my'       O     None   
                11       [49, 53): 'hard'       [49, 53): 'hard'       B      FAB   
                12       [54, 58): 'disk'       [54, 58): 'disk'       B      FAB   
                13          [58, 59): ')'          [58, 59): ')'       O     None   
                14          [59, 60): '?'          [59, 60): '?'       O     None   
                15         [61, 63): 'If'         [61, 63): 'If'       B      PER   
                16     [64, 70): 'Barbie'     [64, 70): 'Barbie'       I      PER   
                17         [71, 73): 'is'         [71, 73): 'is'       O     None   
                18         [74, 76): 'so'         [74, 76): 'so'       O     None   
                19    [77, 84): 'popular'    [77, 84): 'popular'       O     None   
                20          [84, 85): ','          [84, 85): ','       O     None   
                21        [86, 89): 'why'        [86, 89): 'why'       O     None   
                22         [90, 92): 'do'         [90, 92): 'do'       O     None   
                23        [93, 96): 'you'        [93, 96): 'you'       O     None   
                24      [97, 101): 'have'      [97, 101): 'have'       O     None   
                25       [102, 104): 'to'       [102, 104): 'to'       O     None   
                26      [105, 108): 'buy'      [105, 108): 'buy'       O     None   
                27      [109, 112): 'her'      [109, 112): 'her'       O     None   
                28  [113, 120): 'friends'  [113, 120): 'friends'       B      ORG   
                29        [120, 121): '?'        [120, 121): '?'       O     None   
                
                                                             sentence  
                0   [0, 60): 'Who is General Failure (and why is h...  
                1   [0, 60): 'Who is General Failure (and why is h...  
                2   [0, 60): 'Who is General Failure (and why is h...  
                3   [0, 60): 'Who is General Failure (and why is h...  
                4   [0, 60): 'Who is General Failure (and why is h...  
                5   [0, 60): 'Who is General Failure (and why is h...  
                6   [0, 60): 'Who is General Failure (and why is h...  
                7   [0, 60): 'Who is General Failure (and why is h...  
                8   [0, 60): 'Who is General Failure (and why is h...  
                9   [0, 60): 'Who is General Failure (and why is h...  
                10  [0, 60): 'Who is General Failure (and why is h...  
                11  [0, 60): 'Who is General Failure (and why is h...  
                12  [0, 60): 'Who is General Failure (and why is h...  
                13  [0, 60): 'Who is General Failure (and why is h...  
                14  [0, 60): 'Who is General Failure (and why is h...  
                15  [61, 121): 'If Barbie is so popular, why do yo...  
                16  [61, 121): 'If Barbie is so popular, why do yo...  
                17  [61, 121): 'If Barbie is so popular, why do yo...  
                18  [61, 121): 'If Barbie is so popular, why do yo...  
                19  [61, 121): 'If Barbie is so popular, why do yo...  
                20  [61, 121): 'If Barbie is so popular, why do yo...  
                21  [61, 121): 'If Barbie is so popular, why do yo...  
                22  [61, 121): 'If Barbie is so popular, why do yo...  
                23  [61, 121): 'If Barbie is so popular, why do yo...  
                24  [61, 121): 'If Barbie is so popular, why do yo...  
                25  [61, 121): 'If Barbie is so popular, why do yo...  
                26  [61, 121): 'If Barbie is so popular, why do yo...  
                27  [61, 121): 'If Barbie is so popular, why do yo...  
                28  [61, 121): 'If Barbie is so popular, why do yo...  
                29  [61, 121): 'If Barbie is so popular, why do yo...  """
            ),
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
            ),
        )

    def test_add_token_classes(self):
        df = make_tokens_and_features(
            textwrap.dedent(
                """\
                I had amnesia once or twice.
                -- Steven Wright"""
            ),
            _SPACY_LANGUAGE_MODEL,
        )
        df_with_labels = add_token_classes(df)
        relevant_cols = df_with_labels[["char_span", "token_class", "token_class_id"]]
        # print(f"****{relevant_cols}****")
        self.assertEqual(
            str(relevant_cols),
            textwrap.dedent(
                """\
                         char_span token_class  token_class_id
            0          [0, 1): 'I'           O               0
            1        [2, 5): 'had'           O               0
            2   [6, 13): 'amnesia'           O               0
            3     [14, 18): 'once'           O               0
            4       [19, 21): 'or'           O               0
            5    [22, 27): 'twice'           O               0
            6        [27, 28): '.'           O               0
            7         [28, 29): ''           O               0
            8       [29, 31): '--'           O               0
            9   [32, 38): 'Steven'    B-PERSON               1
            10  [39, 45): 'Wright'    I-PERSON               2"""
            ),
        )


if __name__ == "__main__":
    unittest.main()
