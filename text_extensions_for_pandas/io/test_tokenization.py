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
from transformers import BertTokenizerFast

from text_extensions_for_pandas.io.tokenization import make_bert_tokens


class TestTokenize(unittest.TestCase):

    def setUp(self):
        # Ensure that diffs are consistent
        pd.set_option("display.max_columns", 250)

    def tearDown(self):
        pd.reset_option("display.max_columns")

    def test_make_bert_tokens(self):
        text = "the quick brown fox jumps over the lazy dog."
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', add_special_tokens=True)
        bert_toks = make_bert_tokens(text, tokenizer)

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
