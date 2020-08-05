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

import unittest

import pandas as pd
import numpy as np
import textwrap

# noinspection PyPackageRequirements
from transformers import BertTokenizerFast, BertModel

from text_extensions_for_pandas.io.tokenization import (
    make_bert_tokens, windows_to_seq, seq_to_windows, add_embeddings, conll_to_bert
)
from text_extensions_for_pandas.io.conll import (
    conll_2003_to_dataframes, make_iob_tag_categories
)


class TestTokenize(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTokenize, self).__init__(*args, **kwargs)
        # Instantiate expensive-to-load models once
        model_name = "bert-base-uncased"
        self._tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                            add_special_tokens=True)
        self._bert = BertModel.from_pretrained(model_name)

    def setUp(self):
        # Ensure that diffs are consistent
        pd.set_option("display.max_columns", 250)

    def tearDown(self):
        pd.reset_option("display.max_columns")

    def test_make_bert_tokens(self):
        text = "the quick brown fox jumps over the lazy dog."
        bert_toks = make_bert_tokens(text, self._tokenizer)

        self.assertIsInstance(bert_toks, pd.DataFrame)
        self.assertEqual(len(bert_toks.columns), 7)

        # verify input id is correct
        self.assertTrue('input_id' in bert_toks.columns)
        input_id = bert_toks['input_id']
        self.assertEqual(input_id.iloc[0], 101)  # docstart
        self.assertEqual(input_id.iloc[len(input_id) - 1], 102)  # docend
        self.assertEqual(len(set(input_id)), len(input_id) - 1)  # token "the" appears twice

        # verify special tokens mask
        self.assertTrue('special_tokens_mask' in bert_toks.columns)
        special_tokens_mask = bert_toks['special_tokens_mask']
        self.assertEqual(special_tokens_mask.iloc[0], True)
        self.assertTrue(not any(special_tokens_mask.iloc[1:-1]))
        self.assertEqual(special_tokens_mask.iloc[len(special_tokens_mask) - 1], True)

        # verify char span array
        self.assertTrue('char_span' in bert_toks.columns)
        char_span = bert_toks['char_span']
        span = char_span.iloc[0]
        self.assertEqual(span.begin, span.end)  # docstart special token
        span = char_span.iloc[len(char_span) - 1]
        self.assertEqual(span.begin, span.end)  # docend special token
        expected = text[:-1].split() + [text[-1]]  # remove then append period
        tokens = [span.covered_text for span in char_span]
        for e in expected:
            self.assertTrue(e in tokens)

        # verify token span array
        self.assertTrue('token_span' in bert_toks.columns)
        token_span = bert_toks['token_span']
        span = token_span.iloc[0]
        self.assertEqual(span.begin, span.end)  # docstart special token
        span = token_span.iloc[len(token_span) - 1]
        self.assertEqual(span.begin, span.end)  # docend special token

        # verify token spans match char spans
        char_span_text = [s.covered_text for s in char_span]
        token_span_text = [s.covered_text for s in token_span]
        self.assertListEqual(char_span_text, token_span_text)

    def test_add_embeddings(self):
        text = "What's another word for Thesaurus?"
        bert_toks = make_bert_tokens(text, self._tokenizer)
        toks_with_embeddings = add_embeddings(bert_toks, self._bert)
        # print(f"***{toks_with_embeddings}***")
        self.assertEqual(
            str(toks_with_embeddings),
            # NOTE: Don't forget to add both sets of double-backslashes back in if you
            # copy-and-paste an updated version of the output below!
            textwrap.dedent(
                """\
                id           char_span          token_span  input_id  token_type_id  \\
            0    0          [0, 0): ''          [0, 0): ''       101              0   
            1    1      [0, 4): 'What'      [0, 4): 'What'      2054              0   
            2    2         [4, 5): '''         [4, 5): '''      1005              0   
            3    3         [5, 6): 's'         [5, 6): 's'      1055              0   
            4    4  [7, 14): 'another'  [7, 14): 'another'      2178              0   
            5    5    [15, 19): 'word'    [15, 19): 'word'      2773              0   
            6    6     [20, 23): 'for'     [20, 23): 'for'      2005              0   
            7    7     [24, 27): 'The'     [24, 27): 'The'      1996              0   
            8    8  [27, 33): 'saurus'  [27, 33): 'saurus'     22244              0   
            9    9       [33, 34): '?'       [33, 34): '?'      1029              0   
            10  10          [0, 0): ''          [0, 0): ''       102              0   
            
                attention_mask  special_tokens_mask  \\
            0                1                 True   
            1                1                False   
            2                1                False   
            3                1                False   
            4                1                False   
            5                1                False   
            6                1                False   
            7                1                False   
            8                1                False   
            9                1                False   
            10               1                 True   
            
                                                        embedding  
            0  [-1.28466532e-01  1.48239672e-01  2.22291630e-0...  
            1  [ 1.25274494e-01 -9.84855667e-02  9.63621438e-0...  
            2  [ 4.06849980e-01  1.57153994e-01  6.80696249e-0...  
            3  [ 8.79898727e-01  1.75505936e-01  4.46715385e-0...  
            4  [ 6.10884547e-01 -5.06082773e-01  2.02258557e-0...  
            5  [ 2.25633755e-01  1.60800070e-01 -7.63645768e-0...  
            6  [ 3.99375081e-01  5.56941926e-01  6.83342814e-0...  
            7  [-8.26297551e-02 -6.08669817e-01  1.29478961e-0...  
            8  [ 1.20107301e-01 -2.46813118e-01  1.61801279e-0...  
            9  [ 9.03995410e-02 -2.36030549e-01 -5.32582641e-0...  
            10 [ 7.81801343e-01  2.77477980e-01 -9.42935795e-0...  """))

    def test_conll_to_bert(self):
        dfs = conll_2003_to_dataframes("test_data/io/test_conll/conll03_test.txt",
                                       ["ent"], [True])
        first_df = dfs[0]
        token_class_dtype, int_to_label, label_to_int = make_iob_tag_categories(
            ["MISC", "PER", "FOO", "BAR", "FAB"])
        with_embeddings = conll_to_bert(
            first_df, self._tokenizer, self._bert, token_class_dtype,
            compute_embeddings=True)
        num_rows = 5
        # print(f"[[[{with_embeddings.iloc[:num_rows]}]]]")
        self.assertEqual(
            str(with_embeddings.iloc[:num_rows]),
            # NOTE: Don't forget to add both sets of double-backslashes back in if you
            # copy-and-paste an updated version of the output below!
            textwrap.dedent("""\
               id            char_span           token_span  input_id  token_type_id  \\
            0   0           [0, 0): ''           [0, 0): ''       101              0   
            1   1        [0, 3): 'Who'        [0, 3): 'Who'      2040              0   
            2   2         [4, 6): 'is'         [4, 6): 'is'      2003              0   
            3   3   [7, 14): 'General'   [7, 14): 'General'      2236              0   
            4   4  [15, 22): 'Failure'  [15, 22): 'Failure'      4945              0   
            
               attention_mask  special_tokens_mask ent_iob ent_type token_class  \\
            0               1                 True       O     <NA>           O   
            1               1                False       O     <NA>           O   
            2               1                False       O     <NA>           O   
            3               1                False       B      PER       B-PER   
            4               1                False       I      PER       I-PER   
            
               token_class_id                                          embedding  
            0               0 [-5.97730204e-02  1.62091315e-01  3.69775221e-0...  
            1               0 [-1.47429836e+00  2.81369209e-01  3.91599715e-0...  
            2               0 [-5.15211642e-01  1.88511163e-01  4.33955133e-0...  
            3               2 [-4.91548181e-01 -6.86507523e-02  5.30912206e-0...  
            4               7 [-1.64590195e-01  4.79323864e-02  4.46191430e-0...  """))

        without_embeddings = conll_to_bert(
            first_df, self._tokenizer, self._bert, token_class_dtype,
            compute_embeddings=False)
        # print(f"[[[{without_embeddings.iloc[:num_rows]}]]]")
        self.assertEqual(
            str(without_embeddings.iloc[:num_rows]),
            # NOTE: Don't forget to add both sets of double-backslashes back in if you
            # copy-and-paste an updated version of the output below!
            textwrap.dedent("""\
               id            char_span           token_span  input_id  token_type_id  \\
            0   0           [0, 0): ''           [0, 0): ''       101              0   
            1   1        [0, 3): 'Who'        [0, 3): 'Who'      2040              0   
            2   2         [4, 6): 'is'         [4, 6): 'is'      2003              0   
            3   3   [7, 14): 'General'   [7, 14): 'General'      2236              0   
            4   4  [15, 22): 'Failure'  [15, 22): 'Failure'      4945              0   
            
               attention_mask  special_tokens_mask ent_iob ent_type token_class  \\
            0               1                 True       O     <NA>           O   
            1               1                False       O     <NA>           O   
            2               1                False       O     <NA>           O   
            3               1                False       B      PER       B-PER   
            4               1                False       I      PER       I-PER   
            
               token_class_id  
            0               0  
            1               0  
            2               0  
            3               2  
            4               7  """))

    def test_seq_to_windows(self):
        for seqlen in range(1, 20):
            seq = np.arange(1, seqlen)
            seq_after = windows_to_seq(seq, seq_to_windows(seq, 2, 3)["input_ids"], 2, 3)
            if np.any(seq != seq_after):
                raise ValueError("Before: {seq}; After: {seq_after}")

        for seqlen in range(200, 400):
            seq = np.arange(1, seqlen)
            windows = seq_to_windows(seq, 32, 64)
            seq_after = windows_to_seq(seq, windows["input_ids"], 32, 64)
            if np.any(seq != seq_after):
                raise ValueError("Before: {seq}; After: {seq_after}")

        for seqlen in range(50, 100):
            seq = np.arange(1, seqlen)
            windows = seq_to_windows(seq, 32, 64)
            seq_after = windows_to_seq(seq, windows["input_ids"], 32, 64)
            if np.any(seq != seq_after):
                raise ValueError("Before: {seq}; After: {seq_after}")
