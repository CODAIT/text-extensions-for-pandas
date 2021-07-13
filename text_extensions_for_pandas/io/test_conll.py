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
import os
import shutil

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
                                          span ent_type
                0           [61, 67): 'Alaska'      GPE
                1      [73, 84): 'Santa Claus'      ORG
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
        self.assertTrue("span" in spans.columns)
        result = spans_to_iob(spans["span"])
        pd.testing.assert_series_equal(df["ent_iob"], result["ent_iob"])

    def test_conll_2003_to_dataframes(self):
        dfs = conll_2003_to_dataframes(
            "test_data/io/test_conll/conll03_test.txt", ["ent"], [True]
        )
        self.assertEqual(len(dfs), 2)
        self.assertEqual(
            dfs[0]["span"].values.document_text,
            textwrap.dedent(
                """\
                Who is General Failure (and why is he reading my hard disk)?
                If Barbie is so popular, why do you have to buy Barbie's friends?"""
            ),
        )
        self.assertEqual(
            dfs[1]["span"].values.document_text,
            "-DOCSTART-\nI'd kill for a Nobel Peace Prize.",
        )
        # print(f"***{repr(dfs[0])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(dfs[0]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                     span ent_iob ent_type  \\
                0           [0, 3): 'Who'       O     None   
                1            [4, 6): 'is'       O     None   
                2      [7, 14): 'General'       B      PER   
                3     [15, 22): 'Failure'       I      PER   
                4           [23, 24): '('       O     None   
                5         [24, 27): 'and'       O     None   
                6         [28, 31): 'why'       O     None   
                7          [32, 34): 'is'       B      FOO   
                8          [35, 37): 'he'       B      BAR   
                9     [38, 45): 'reading'       O     None   
                10         [46, 48): 'my'       O     None   
                11       [49, 53): 'hard'       B      FAB   
                12       [54, 58): 'disk'       B      FAB   
                13          [58, 59): ')'       O     None   
                14          [59, 60): '?'       O     None   
                15         [61, 63): 'If'       O     None   
                16     [64, 70): 'Barbie'       B      PER   
                17         [71, 73): 'is'       O     None   
                18         [74, 76): 'so'       O     None   
                19    [77, 84): 'popular'       O     None   
                20          [84, 85): ','       O     None   
                21        [86, 89): 'why'       O     None   
                22         [90, 92): 'do'       O     None   
                23        [93, 96): 'you'       O     None   
                24      [97, 101): 'have'       O     None   
                25       [102, 104): 'to'       O     None   
                26      [105, 108): 'buy'       O     None   
                27   [109, 115): 'Barbie'       B      PER   
                28       [115, 117): ''s'       O     None   
                29  [118, 125): 'friends'       O     None   
                30        [125, 126): '?'       O     None   
                
                                                             sentence  line_num  
                0   [0, 60): 'Who is General Failure (and why is h...         0  
                1   [0, 60): 'Who is General Failure (and why is h...         1  
                2   [0, 60): 'Who is General Failure (and why is h...         2  
                3   [0, 60): 'Who is General Failure (and why is h...         3  
                4   [0, 60): 'Who is General Failure (and why is h...         4  
                5   [0, 60): 'Who is General Failure (and why is h...         5  
                6   [0, 60): 'Who is General Failure (and why is h...         6  
                7   [0, 60): 'Who is General Failure (and why is h...         7  
                8   [0, 60): 'Who is General Failure (and why is h...         8  
                9   [0, 60): 'Who is General Failure (and why is h...         9  
                10  [0, 60): 'Who is General Failure (and why is h...        10  
                11  [0, 60): 'Who is General Failure (and why is h...        11  
                12  [0, 60): 'Who is General Failure (and why is h...        12  
                13  [0, 60): 'Who is General Failure (and why is h...        13  
                14  [0, 60): 'Who is General Failure (and why is h...        14  
                15  [61, 126): 'If Barbie is so popular, why do yo...        16  
                16  [61, 126): 'If Barbie is so popular, why do yo...        17  
                17  [61, 126): 'If Barbie is so popular, why do yo...        18  
                18  [61, 126): 'If Barbie is so popular, why do yo...        19  
                19  [61, 126): 'If Barbie is so popular, why do yo...        20  
                20  [61, 126): 'If Barbie is so popular, why do yo...        21  
                21  [61, 126): 'If Barbie is so popular, why do yo...        22  
                22  [61, 126): 'If Barbie is so popular, why do yo...        23  
                23  [61, 126): 'If Barbie is so popular, why do yo...        24  
                24  [61, 126): 'If Barbie is so popular, why do yo...        25  
                25  [61, 126): 'If Barbie is so popular, why do yo...        26  
                26  [61, 126): 'If Barbie is so popular, why do yo...        27  
                27  [61, 126): 'If Barbie is so popular, why do yo...        28  
                28  [61, 126): 'If Barbie is so popular, why do yo...        29  
                29  [61, 126): 'If Barbie is so popular, why do yo...        30  
                30  [61, 126): 'If Barbie is so popular, why do yo...        31  """
            ),
        )

    def test_conll_2003_to_dataframes_multi_field(self):
        dfs = conll_2003_to_dataframes(
            "test_data/io/test_conll/conll03_test2.txt",
            ["pos", "phrase", "ent"],
            [False, True, True],
        )
        # print(f"***{repr(dfs[0])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(dfs[0]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                     span  pos phrase_iob phrase_type ent_iob ent_type  \\
                0   [0, 10): '-DOCSTART-'  -X-          O        None       O     None   
                1         [11, 14): 'Who'   WP          B          NP       O     None   
                2          [15, 17): 'is'  VBD          B          VP       O     None   
                3     [18, 25): 'General'  NNP          B          NP       B      PER   
                4     [26, 33): 'Failure'  NNP          B          NP       I      PER   
                5           [34, 35): '('    (          O        None       O     None   
                6         [35, 38): 'and'   CC          O        None       O     None   
                7         [39, 42): 'why'  WRB          B        ADVP       O     None   
                8          [43, 45): 'is'  VPD          B          VP       B      FOO   
                9          [46, 48): 'he'  PRP          B          NP       B      BAR   
                10    [49, 56): 'reading'  VBD          B          VP       O     None   
                11         [57, 59): 'my'  WRB          I          VP       O     None   
                12       [60, 64): 'hard'   NN          I          VP       B      FAB   
                13       [65, 69): 'disk'   NN          I          VP       B      FAB   
                14          [69, 70): ')'    )          O        None       O     None   
                15          [70, 71): '?'    ?          O        None       O     None   
                16         [72, 74): 'If'   CC          O        None       O     None   
                17     [75, 81): 'Barbie'  NNP          B          NP       B      PER   
                18         [82, 84): 'is'  VPD          B          VP       O     None   
                19         [85, 87): 'so'  WRB          B        ADJP       O     None   
                20    [88, 95): 'popular'   JJ          I        ADJP       O     None   
                21          [95, 96): ','    ,          O        None       O     None   
                22       [97, 100): 'why'  WRB          B        ADVP       O     None   
                23       [101, 103): 'do'  VPD          B          VP       O     None   
                24      [104, 107): 'you'   NN          O        None       O     None   
                25     [108, 112): 'have'  VBD          B          VP       O     None   
                26       [113, 115): 'to'  VBD          I          VP       O     None   
                27      [116, 119): 'buy'  VBD          I          VP       O     None   
                28   [120, 126): 'Barbie'  NNP          B          NP       B      PER   
                29       [126, 128): ''s'    '          O        None       O     None   
                30  [129, 136): 'friends'   NN          B          NP       O     None   
                31        [136, 137): '?'    ?          O        None       O     None   
                
                                                             sentence  line_num  
                0                               [0, 10): '-DOCSTART-'         0  
                1   [11, 71): 'Who is General Failure (and why is ...         2  
                2   [11, 71): 'Who is General Failure (and why is ...         3  
                3   [11, 71): 'Who is General Failure (and why is ...         4  
                4   [11, 71): 'Who is General Failure (and why is ...         5  
                5   [11, 71): 'Who is General Failure (and why is ...         6  
                6   [11, 71): 'Who is General Failure (and why is ...         7  
                7   [11, 71): 'Who is General Failure (and why is ...         8  
                8   [11, 71): 'Who is General Failure (and why is ...         9  
                9   [11, 71): 'Who is General Failure (and why is ...        10  
                10  [11, 71): 'Who is General Failure (and why is ...        11  
                11  [11, 71): 'Who is General Failure (and why is ...        12  
                12  [11, 71): 'Who is General Failure (and why is ...        13  
                13  [11, 71): 'Who is General Failure (and why is ...        14  
                14  [11, 71): 'Who is General Failure (and why is ...        15  
                15  [11, 71): 'Who is General Failure (and why is ...        16  
                16  [72, 137): 'If Barbie is so popular, why do yo...        18  
                17  [72, 137): 'If Barbie is so popular, why do yo...        19  
                18  [72, 137): 'If Barbie is so popular, why do yo...        20  
                19  [72, 137): 'If Barbie is so popular, why do yo...        21  
                20  [72, 137): 'If Barbie is so popular, why do yo...        22  
                21  [72, 137): 'If Barbie is so popular, why do yo...        23  
                22  [72, 137): 'If Barbie is so popular, why do yo...        24  
                23  [72, 137): 'If Barbie is so popular, why do yo...        25  
                24  [72, 137): 'If Barbie is so popular, why do yo...        26  
                25  [72, 137): 'If Barbie is so popular, why do yo...        27  
                26  [72, 137): 'If Barbie is so popular, why do yo...        28  
                27  [72, 137): 'If Barbie is so popular, why do yo...        29  
                28  [72, 137): 'If Barbie is so popular, why do yo...        30  
                29  [72, 137): 'If Barbie is so popular, why do yo...        31  
                30  [72, 137): 'If Barbie is so popular, why do yo...        32  
                31  [72, 137): 'If Barbie is so popular, why do yo...        33  """
            ),
        )

    def test_conll_u_to_dataframes(self):
        dfs = conll_u_to_dataframes("test_data/io/test_conll/conll_u_test1.txt")
        self.maxDiff = None
        print(f"***{repr(dfs[3])}***")
        self.assertEqual(
            repr(dfs[3]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                        span      lemma upostag xpostag  \\
                0           [0, 6): 'Google'     Google   PROPN     NNP   
                1             [7, 10): 'has'       have     AUX     VBZ   
                2        [11, 18): 'finally'    finally     ADV      RB   
                3            [19, 22): 'had'       have    VERB     VBN   
                4             [23, 25): 'an'          a     DET      DT   
                ..                       ...        ...     ...     ...   
                161          [776, 777): 'a'          a     DET      DT   
                162  [778, 787): 'punchline'  punchline    NOUN      NN   
                163          [787, 788): ','          ,   PUNCT       ,   
                164        [789, 792): 'too'        too     ADV      RB   
                165          [792, 793): '.'          .   PUNCT       .   
                
                                                              features  head  deprel  \\
                0                                          Number=Sing     3   nsubj   
                1    Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbF...     3     aux   
                2                                                 None     3  advmod   
                3                             Tense=Past|VerbForm=Part  <NA>    root   
                4                            Definite=Ind|PronType=Art     6     det   
                ..                                                 ...   ...     ...   
                161                          Definite=Ind|PronType=Art   162     det   
                162                                        Number=Sing   160   nsubj   
                163                                               None   160   punct   
                164                                               None   160  advmod   
                165                                               None   154   punct   
                
                         deps           misc  \\
                0     4:nsubj           None   
                1       4:aux           None   
                2    4:advmod           None   
                3      0:root           None   
                4       7:det           None   
                ..        ...            ...   
                161     9:det           None   
                162   7:nsubj  SpaceAfter=No   
                163   7:punct           None   
                164  7:advmod  SpaceAfter=No   
                165   1:punct           None   
                
                                                           sentence_id  \\
                0    weblog-blogspot.com_marketview_20050210075500_...   
                1    weblog-blogspot.com_marketview_20050210075500_...   
                2    weblog-blogspot.com_marketview_20050210075500_...   
                3    weblog-blogspot.com_marketview_20050210075500_...   
                4    weblog-blogspot.com_marketview_20050210075500_...   
                ..                                                 ...   
                161  weblog-blogspot.com_marketview_20050210075500_...   
                162  weblog-blogspot.com_marketview_20050210075500_...   
                163  weblog-blogspot.com_marketview_20050210075500_...   
                164  weblog-blogspot.com_marketview_20050210075500_...   
                165  weblog-blogspot.com_marketview_20050210075500_...   
                
                                                          paragraph_id  \\
                0    weblog-blogspot.com_marketview_20050210075500_...   
                1    weblog-blogspot.com_marketview_20050210075500_...   
                2    weblog-blogspot.com_marketview_20050210075500_...   
                3    weblog-blogspot.com_marketview_20050210075500_...   
                4    weblog-blogspot.com_marketview_20050210075500_...   
                ..                                                 ...   
                161  weblog-blogspot.com_marketview_20050210075500_...   
                162  weblog-blogspot.com_marketview_20050210075500_...   
                163  weblog-blogspot.com_marketview_20050210075500_...   
                164  weblog-blogspot.com_marketview_20050210075500_...   
                165  weblog-blogspot.com_marketview_20050210075500_...   
                
                                                                doc_id  \\
                0    weblog-blogspot.com_marketview_20050210075500_...   
                1    weblog-blogspot.com_marketview_20050210075500_...   
                2    weblog-blogspot.com_marketview_20050210075500_...   
                3    weblog-blogspot.com_marketview_20050210075500_...   
                4    weblog-blogspot.com_marketview_20050210075500_...   
                ..                                                 ...   
                161  weblog-blogspot.com_marketview_20050210075500_...   
                162  weblog-blogspot.com_marketview_20050210075500_...   
                163  weblog-blogspot.com_marketview_20050210075500_...   
                164  weblog-blogspot.com_marketview_20050210075500_...   
                165  weblog-blogspot.com_marketview_20050210075500_...   
                
                                                              sentence  line_num  
                0    [0, 139): 'Google has finally had an analyst d...       383  
                1    [0, 139): 'Google has finally had an analyst d...       384  
                2    [0, 139): 'Google has finally had an analyst d...       385  
                3    [0, 139): 'Google has finally had an analyst d...       386  
                4    [0, 139): 'Google has finally had an analyst d...       387  
                ..                                                 ...       ...  
                161  [743, 793): 'Read the entire article; there's ...       565  
                162  [743, 793): 'Read the entire article; there's ...       566  
                163  [743, 793): 'Read the entire article; there's ...       567  
                164  [743, 793): 'Read the entire article; there's ...       568  
                165  [743, 793): 'Read the entire article; there's ...       569  
                
                [166 rows x 14 columns]"""
            ),
        )

        print(f"***{repr(dfs[0])}***")  # catch bug where first df isn't the same
        self.assertEqual(
            repr(dfs[0]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                       span     lemma upostag xpostag  \\
                0            [0, 4): 'From'      from     ADP      IN   
                1             [5, 8): 'the'       the     DET      DT   
                2             [9, 11): 'AP'        AP   PROPN     NNP   
                3         [12, 17): 'comes'      come    VERB     VBZ   
                4          [18, 22): 'this'      this     DET      DT   
                ..                      ...       ...     ...     ...   
                81  [439, 448): 'replacing'   replace    VERB     VBG   
                82       [449, 453): 'John'      John   PROPN     NNP   
                83   [454, 462): 'Montague'  Montague   PROPN     NNP   
                84   [463, 471): 'Steadman'  Steadman   PROPN     NNP   
                85          [471, 472): '.'         .   PUNCT       .   
                
                                                             features  head deprel  \\
                0                                                None     2   case   
                1                           Definite=Def|PronType=Art     2    det   
                2                                         Number=Sing     3    obl   
                3   Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbF...  <NA>   root   
                4                            Number=Sing|PronType=Dem     5    det   
                ..                                                ...   ...    ...   
                81                                       VerbForm=Ger    58  advcl   
                82                                        Number=Sing    81    obj   
                83                                        Number=Sing    82   flat   
                84                                        Number=Sing    82   flat   
                85                                               None    58  punct   
                
                          deps           misc  \\
                0       3:case           None   
                1        3:det           None   
                2   4:obl:from           None   
                3       0:root           None   
                4        6:det           None   
                ..         ...            ...   
                81     3:advcl           None   
                82      26:obj           None   
                83     27:flat           None   
                84     27:flat  SpaceAfter=No   
                85     3:punct           None   
                
                                                          sentence_id  \\
                0   weblog-blogspot.com_nominations_20041117172713...   
                1   weblog-blogspot.com_nominations_20041117172713...   
                2   weblog-blogspot.com_nominations_20041117172713...   
                3   weblog-blogspot.com_nominations_20041117172713...   
                4   weblog-blogspot.com_nominations_20041117172713...   
                ..                                                ...   
                81  weblog-blogspot.com_nominations_20041117172713...   
                82  weblog-blogspot.com_nominations_20041117172713...   
                83  weblog-blogspot.com_nominations_20041117172713...   
                84  weblog-blogspot.com_nominations_20041117172713...   
                85  weblog-blogspot.com_nominations_20041117172713...   
                
                                                         paragraph_id  \\
                0   weblog-blogspot.com_nominations_20041117172713...   
                1   weblog-blogspot.com_nominations_20041117172713...   
                2   weblog-blogspot.com_nominations_20041117172713...   
                3   weblog-blogspot.com_nominations_20041117172713...   
                4   weblog-blogspot.com_nominations_20041117172713...   
                ..                                                ...   
                81  weblog-blogspot.com_nominations_20041117172713...   
                82  weblog-blogspot.com_nominations_20041117172713...   
                83  weblog-blogspot.com_nominations_20041117172713...   
                84  weblog-blogspot.com_nominations_20041117172713...   
                85  weblog-blogspot.com_nominations_20041117172713...   
                
                                                               doc_id  \\
                0   weblog-blogspot.com_nominations_20041117172713...   
                1   weblog-blogspot.com_nominations_20041117172713...   
                2   weblog-blogspot.com_nominations_20041117172713...   
                3   weblog-blogspot.com_nominations_20041117172713...   
                4   weblog-blogspot.com_nominations_20041117172713...   
                ..                                                ...   
                81  weblog-blogspot.com_nominations_20041117172713...   
                82  weblog-blogspot.com_nominations_20041117172713...   
                83  weblog-blogspot.com_nominations_20041117172713...   
                84  weblog-blogspot.com_nominations_20041117172713...   
                85  weblog-blogspot.com_nominations_20041117172713...   
                
                                                             sentence  line_num  
                0            [0, 29): 'From the AP comes this story:'         4  
                1            [0, 29): 'From the AP comes this story:'         5  
                2            [0, 29): 'From the AP comes this story:'         6  
                3            [0, 29): 'From the AP comes this story:'         7  
                4            [0, 29): 'From the AP comes this story:'         8  
                ..                                                ...       ...  
                81  [310, 472): 'Bush also nominated A. Noel Anket...        98  
                82  [310, 472): 'Bush also nominated A. Noel Anket...        99  
                83  [310, 472): 'Bush also nominated A. Noel Anket...       100  
                84  [310, 472): 'Bush also nominated A. Noel Anket...       101  
                85  [310, 472): 'Bush also nominated A. Noel Anket...       102  
                
                [86 rows x 14 columns]"""
            ),
        )
        dfs = conll_u_to_dataframes("test_data/io/test_conll/conll_09_test1.conllu")
        print(f"***{repr(dfs[0])}***")  # catch bug where first df isn't the same
        self.assertEqual(
            repr(dfs[0]),
            textwrap.dedent(
                """\
                                     span   lemma upostag xpostag features  head deprel  deps  \\
                0            [0, 2): 'No'      no      DT      DT     None     3    DEP  None   
                1             [2, 3): ','       ,       ,       ,     None     3      P  None   
                2            [4, 6): 'it'      it     PRP     PRP     None     3    SBJ  None   
                3          [7, 10): 'was'      be     VBD     VBD     None  <NA>   ROOT  None   
                4         [11, 14): 'n't'     not      RB      RB     None     3    ADV  None   
                ..                    ...     ...     ...     ...      ...   ...    ...   ...   
                74     [373, 377): 'both'    both      DT      DT     None    75   NMOD  None   
                75   [378, 384): 'stocks'  stocks     NNS     NNS     None    73   PMOD  None   
                76      [385, 388): 'and'     and      CC      CC     None    75  COORD  None   
                77  [389, 396): 'futures'  future     NNS     NNS     None    76   CONJ  None   
                78        [396, 397): '.'       .       .       .     None    59      P  None   
                
                    misc predicate pred0arg pred1arg pred2arg pred3arg pred4arg pred5arg  \\
                0   None      None     None     None     None     None     None     None   
                1   None      None     None     None     None     None     None     None   
                2   None      None     None     None     None     None     None     None   
                3   None      None     None     None     None     None     None     None   
                4   None      None     None     None     None     None     None     None   
                ..   ...       ...      ...      ...      ...      ...      ...      ...   
                74  None      None     None     None     None     None     None     None   
                75  None      None     None     None     None     None     None     None   
                76  None      None     None     None     None     None     None     None   
                77  None      None     None     None     None     None     None     None   
                78  None      None     None     None     None     None     None     None   
                
                   pred6arg pred7arg                                           sentence  \\
                0      None     None            [0, 28): 'No, it was n't Black Monday.'   
                1      None     None            [0, 28): 'No, it was n't Black Monday.'   
                2      None     None            [0, 28): 'No, it was n't Black Monday.'   
                3      None     None            [0, 28): 'No, it was n't Black Monday.'   
                4      None     None            [0, 28): 'No, it was n't Black Monday.'   
                ..      ...      ...                                                ...   
                74     None     None  [232, 397): 'Some `` circuit breakers '' insta...   
                75     None     None  [232, 397): 'Some `` circuit breakers '' insta...   
                76     None     None  [232, 397): 'Some `` circuit breakers '' insta...   
                77     None     None  [232, 397): 'Some `` circuit breakers '' insta...   
                78     None     None  [232, 397): 'Some `` circuit breakers '' insta...   
                
                    line_num  
                0          1  
                1          2  
                2          3  
                3          4  
                4          5  
                ..       ...  
                74        79  
                75        80  
                76        81  
                77        82  
                78        83  
                
                [79 rows x 20 columns]"""
            ),
        )

    def test_conll_2003_output_to_dataframes(self):
        doc_dfs = conll_2003_to_dataframes(
            "test_data/io/test_conll/conll03_test.txt", ["ent"], [True]
        )
        output_dfs = conll_2003_output_to_dataframes(
            doc_dfs, "test_data/io/test_conll/conll03_output.txt"
        )
        self.assertEqual(len(output_dfs), 2)
        self.assertEqual(
            output_dfs[0]["span"].values.document_text,
            textwrap.dedent(
                """\
                Who is General Failure (and why is he reading my hard disk)?
                If Barbie is so popular, why do you have to buy Barbie's friends?"""
            ),
        )
        self.assertEqual(
            output_dfs[1]["span"].values.document_text,
            "-DOCSTART-\nI'd kill for a Nobel Peace Prize.",
        )
        # print(f"***{repr(output_dfs[0])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(output_dfs[0]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                     span ent_iob ent_type  \\
                0           [0, 3): 'Who'       B     BAND   
                1            [4, 6): 'is'       O     None   
                2      [7, 14): 'General'       B      PER   
                3     [15, 22): 'Failure'       I      PER   
                4           [23, 24): '('       O     None   
                5         [24, 27): 'and'       O     None   
                6         [28, 31): 'why'       B      FOO   
                7          [32, 34): 'is'       I      FOO   
                8          [35, 37): 'he'       B      BAR   
                9     [38, 45): 'reading'       O     None   
                10         [46, 48): 'my'       O     None   
                11       [49, 53): 'hard'       B      FAB   
                12       [54, 58): 'disk'       B      FAB   
                13          [58, 59): ')'       O     None   
                14          [59, 60): '?'       O     None   
                15         [61, 63): 'If'       B      PER   
                16     [64, 70): 'Barbie'       I      PER   
                17         [71, 73): 'is'       O     None   
                18         [74, 76): 'so'       O     None   
                19    [77, 84): 'popular'       O     None   
                20          [84, 85): ','       O     None   
                21        [86, 89): 'why'       O     None   
                22         [90, 92): 'do'       O     None   
                23        [93, 96): 'you'       O     None   
                24      [97, 101): 'have'       O     None   
                25       [102, 104): 'to'       O     None   
                26      [105, 108): 'buy'       O     None   
                27   [109, 115): 'Barbie'       O     None   
                28       [115, 117): ''s'       B      ORG   
                29  [118, 125): 'friends'       O     None   
                30        [125, 126): '?'       O     None   
                
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
                15  [61, 126): 'If Barbie is so popular, why do yo...  
                16  [61, 126): 'If Barbie is so popular, why do yo...  
                17  [61, 126): 'If Barbie is so popular, why do yo...  
                18  [61, 126): 'If Barbie is so popular, why do yo...  
                19  [61, 126): 'If Barbie is so popular, why do yo...  
                20  [61, 126): 'If Barbie is so popular, why do yo...  
                21  [61, 126): 'If Barbie is so popular, why do yo...  
                22  [61, 126): 'If Barbie is so popular, why do yo...  
                23  [61, 126): 'If Barbie is so popular, why do yo...  
                24  [61, 126): 'If Barbie is so popular, why do yo...  
                25  [61, 126): 'If Barbie is so popular, why do yo...  
                26  [61, 126): 'If Barbie is so popular, why do yo...  
                27  [61, 126): 'If Barbie is so popular, why do yo...  
                28  [61, 126): 'If Barbie is so popular, why do yo...  
                29  [61, 126): 'If Barbie is so popular, why do yo...  
                30  [61, 126): 'If Barbie is so popular, why do yo...  """
            ),
        )
        # print(f"***{repr(output_dfs[1])}***")  # Uncomment to regenerate gold standard
        self.assertEqual(
            repr(output_dfs[1]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                    span ent_iob ent_type  \\
                0  [0, 10): '-DOCSTART-'       O     None   
                1        [11, 14): 'I'd'       O     None   
                2       [15, 19): 'kill'       O     None   
                3        [20, 23): 'for'       O     None   
                4          [24, 25): 'a'       O     None   
                5      [26, 31): 'Nobel'       B     MISC   
                6      [32, 37): 'Peace'       O     None   
                7      [38, 43): 'Prize'       B     MISC   
                8          [43, 44): '.'       O     None   
                
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
        relevant_cols = df_with_labels[["span", "token_class", "token_class_id"]]
        # print(f"****{relevant_cols}****")
        self.assertEqual(
            str(relevant_cols),
            textwrap.dedent(
                """\
                              span token_class  token_class_id
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

    def test_combine_folds(self):
        texts = [
            "It's a small world, but I wouldn't want to have to paint it.",
            "Everywhere is walking distance if you have the time.",
        ]
        arrays = [
            SpanArray(texts[0], [7, 20], [12, 23]),
            SpanArray(texts[1], [14], [21]),
        ]
        folds = {
            "train": [pd.DataFrame({"spans": arrays[0], "foos": [1, 2], "bars": True})],
            "bus": [pd.DataFrame({"spans": arrays[1], "foos": [5], "bars": [False]})],
        }
        combined_df = combine_folds(folds)
        # print(f"****{combined_df}****")
        self.assertEqual(
            str(combined_df),
            textwrap.dedent(
                """\
                    fold  doc_num                spans  foos   bars
                0  train        0     [7, 12): 'small'     1   True
                1  train        0      [20, 23): 'but'     2   True
                2    bus        0  [14, 21): 'walking'     5  False"""
            ),
        )
        # Span column should NOT be of object dtype. See issue #73.
        self.assertEqual(str(combined_df["spans"].dtype), "SpanDtype")

    def test_compute_accuracy(self):
        doc_dfs = conll_2003_to_dataframes(
            "test_data/io/test_conll/conll03_test.txt", ["ent"], [True]
        )
        output_dfs = conll_2003_output_to_dataframes(
            doc_dfs, "test_data/io/test_conll/conll03_output.txt"
        )
        stats_by_doc = compute_accuracy_by_document(doc_dfs, output_dfs)
        # print(f"****{stats_by_doc}****")
        self.assertEqual(
            str(stats_by_doc),
            textwrap.dedent(
                """\
          fold  doc_num  num_true_positives  num_extracted  num_entities  precision  \\
        0             0                  24             31            31   0.774194   
        1             1                   7              9             9   0.777778   
        
             recall        F1  
        0  0.774194  0.774194  
        1  0.777778  0.777778  """
            ),
        )
        global_stats = compute_global_accuracy(stats_by_doc)
        # ßprint(f"****{global_stats}****")
        self.assertEqual(
            str(global_stats),
            (
                "{'num_true_positives': 31, 'num_entities': 40, 'num_extracted': 40, "
                "'precision': 0.775, 'recall': 0.775, 'F1': 0.775}"
            ),
        )

    def test_maybe_download_dataset(self):
        base_dir = "test_data/io/test_conll"
        ewt_dir = base_dir + "/ewt_temp"
        conll9_dir = base_dir + "/conll9_temp"
        ewt_url = "https://github.com/UniversalDependencies/UD_English-EWT/blob/master/en_ewt-ud-dev.conllu"
        conll_09_test_data_url = (
            "https://ufal.mff.cuni.cz/conll2009-st/trial/CoNLL2009-ST-English-trial.zip"
        )

        # test download of file
        val = maybe_download_dataset_data(ewt_dir, ewt_url)
        self.assertEqual(val, ewt_dir + "/en_ewt-ud-dev.conllu")
        self.assertTrue(os.path.isdir(ewt_dir))
        self.assertTrue(os.path.isfile(ewt_dir + "/en_ewt-ud-dev.conllu"))

        # test functionality when file already exists
        val = maybe_download_dataset_data(ewt_dir, ewt_url)
        self.assertEqual(val, ewt_dir + "/en_ewt-ud-dev.conllu")
        self.assertTrue(os.path.isdir(ewt_dir))
        self.assertTrue(os.path.isfile(ewt_dir + "/en_ewt-ud-dev.conllu"))

        # test download of alternately named file
        val = maybe_download_dataset_data(ewt_dir, ewt_url, fname="dev.conllu")
        self.assertEqual(val, ewt_dir + "/dev.conllu")
        self.assertTrue(os.path.isdir(ewt_dir))
        self.assertTrue(os.path.isfile(ewt_dir + "/dev.conllu"))
        # check we didn't overwrite the last file
        self.assertTrue(os.path.isfile(ewt_dir + "/en_ewt-ud-dev.conllu"))
        # verify functionality when file already exists
        val = maybe_download_dataset_data(ewt_dir, ewt_url, fname="dev.conllu")
        self.assertEqual(val, ewt_dir + "/dev.conllu")
        self.assertTrue(os.path.isdir(ewt_dir))
        self.assertTrue(os.path.isfile(ewt_dir + "/dev.conllu"))
        # check we didn't overwrite the last file
        self.assertTrue(os.path.isfile(ewt_dir + "/en_ewt-ud-dev.conllu"))


        # test zip
        conll_9_file = conll9_dir + "/CoNLL2009-ST-English-trial.txt"
        val = maybe_download_dataset_data(conll9_dir, conll_09_test_data_url)
        self.assertEqual(val, conll_9_file)
        self.assertTrue(os.path.isdir(conll9_dir))
        self.assertTrue(os.path.isfile(conll_9_file))
        # verify we don't double download for zips
        os.remove(conll9_dir + "/CoNLL2009-ST-English-trial.zip")
        maybe_download_dataset_data(
            conll9_dir,
            conll_09_test_data_url,
            fname="/CoNLL2009-ST-English-trial.txt",
        )
        self.assertFalse(os.path.exists(conll9_dir + "CoNLL2009-ST-English-trial.zip"))

        #clean up by removing our two temp dirs
        shutil.rmtree(ewt_dir)
        shutil.rmtree(conll9_dir)


if __name__ == "__main__":
    unittest.main()
