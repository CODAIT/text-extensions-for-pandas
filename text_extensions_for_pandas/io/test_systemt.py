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

from text_extensions_for_pandas.io import *

import spacy

_SPACY_LANGUAGE_MODEL = spacy.load("en_core_web_sm")


class IOTest(unittest.TestCase):

    def test_load_dict(self):
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


if __name__ == "__main__":
    unittest.main()
