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

import re
import textwrap

from text_extensions_for_pandas.spanner.extract import *
from text_extensions_for_pandas.io.spacy import make_tokens
from text_extensions_for_pandas.util import TestBase

import spacy


class ExtractTest(TestBase):

    @staticmethod
    def _make_tokenizer():
        from spacy.lang.en import English
        nlp = English()
        return nlp.Defaults.create_tokenizer(nlp)

    @staticmethod
    def _read_file(file_name):
        with open(file_name, "r") as f:
            lines = [
                line.strip() for line in f.readlines() if len(line) > 0 and line[0] != "#"
            ]
        return " ".join(lines)

    def test_load_dict(self):
        # noinspection PyPackageRequirements
        from spacy.lang.en import English
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        df = load_dict("test_data/io/test_systemt/test.dict", tokenizer)
        # print(f"***{df}***")
        self.assertEqual(
            str(df),
            textwrap.dedent(
                """\
                       toks_0 toks_1  toks_2   toks_3 toks_4   toks_5 toks_6
                0  dictionary  entry    None     None   None     None   None
                1       entry   None    None     None   None     None   None
                2        help     me       !        i     am  trapped   None
                3          in      a   haiku  factory      !     None   None
                4        save     me  before     they   None     None   None
                5        None   None    None     None   None     None   None"""
            )
        )

    def test_extract_dict(self):
        file_name = "test_data/io/test_systemt/test.dict"
        file_text = self._read_file(file_name)

        tokenizer = self._make_tokenizer()
        char_span = make_tokens(file_text, tokenizer)

        dict_df = load_dict(file_name, tokenizer)

        result_df = extract_dict(char_span, dict_df, "result")

        self.assertIn("result", result_df.columns)
        #print(f"****\n{result_df}\n****")
        self.assertEqual(
            repr(result_df),
            textwrap.dedent(
                """\
                                              result
                2        [0, 16): 'Dictionary Entry'
                0                  [11, 16): 'Entry'
                1                  [17, 22): 'Entry'
                5  [23, 44): 'Help me! I am trapped'
                4    [45, 64): 'In a Haiku factory!'
                3    [65, 84): 'Save me before they'"""
            )
        )

    def test_extract_regex_tok(self):
        file_name = "test_data/io/test_systemt/test.dict"
        file_text = self._read_file(file_name)

        tokenizer = self._make_tokenizer()
        char_span = make_tokens(file_text, tokenizer)

        match_regex = re.compile(r".*y$")

        result_df = extract_regex_tok(char_span, match_regex, output_col_name="result")

        self.assertIn("result", result_df.columns)
        #print(f"****\n{result_df}\n****")
        self.assertEqual(
            repr(result_df),
            textwrap.dedent(
                """\
                                  result
                0  [0, 10): 'Dictionary'
                1      [11, 16): 'Entry'
                2      [17, 22): 'Entry'
                3    [56, 63): 'factory'
                4       [80, 84): 'they'"""
            )
        )

    def test_extract_regex_tok_len_2(self):
        file_name = "test_data/io/test_systemt/test.dict"
        file_text = self._read_file(file_name)

        tokenizer = self._make_tokenizer()
        char_span = make_tokens(file_text, tokenizer)

        match_regex = re.compile(r".*y$")

        result_df = extract_regex_tok(char_span, match_regex, min_len=2, max_len=2,
                                      output_col_name="result")

        self.assertIn("result", result_df.columns)
        #print(f"****\n{result_df}\n****")
        self.assertEqual(
            repr(result_df),
            textwrap.dedent(
                """\
                                        result
                0  [0, 16): 'Dictionary Entry'
                1      [11, 22): 'Entry Entry'
                2    [50, 63): 'Haiku factory'
                3      [73, 84): 'before they'"""
            )
        )
