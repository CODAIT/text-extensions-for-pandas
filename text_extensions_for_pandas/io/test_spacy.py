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

from text_extensions_for_pandas.io.spacy import *

import spacy

_SPACY_LANGUAGE_MODEL = spacy.load("en_core_web_sm")


class IOTest(unittest.TestCase):
    def test_make_tokens(self):
        from spacy.lang.en import English
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        series = make_tokens(
            "The quick, brown fox jumped over the hazy bog...", tokenizer
        )
        self.assertEqual(
            repr(series),
            textwrap.dedent(
                """\
                0          [0, 3): 'The'
                1        [4, 9): 'quick'
                2           [9, 10): ','
                3      [11, 16): 'brown'
                4        [17, 20): 'fox'
                5     [21, 27): 'jumped'
                6       [28, 32): 'over'
                7        [33, 36): 'the'
                8       [37, 41): 'hazy'
                9        [42, 45): 'bog'
                10       [45, 48): '...'
                dtype: SpanDtype"""
            ),
        )

    def test_make_tokens_and_features(self):
        df = make_tokens_and_features(
            "She sold c shills by the Sith Lord.", _SPACY_LANGUAGE_MODEL
        )
        # print(f"****{str(df.to_records())}****")
        self.assertEqual(
            str(df.to_records()),
            textwrap.dedent(
                """\
                [(0, 0, [0, 3): 'She', '-PRON-', 'PRON', 'PRP', 'nsubj', 1, 'Xxx', 'O', '',  True,  True, [0, 35): 'She sold c shills by the Sith Lord.')
                 (1, 1, [4, 8): 'sold', 'sell', 'VERB', 'VBD', 'ROOT', 1, 'xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.')
                 (2, 2, [9, 10): 'c', 'c', 'NOUN', 'NN', 'det', 3, 'x', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.')
                 (3, 3, [11, 17): 'shills', 'shill', 'NOUN', 'NNS', 'dobj', 1, 'xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.')
                 (4, 4, [18, 20): 'by', 'by', 'ADP', 'IN', 'prep', 3, 'xx', 'O', '',  True,  True, [0, 35): 'She sold c shills by the Sith Lord.')
                 (5, 5, [21, 24): 'the', 'the', 'DET', 'DT', 'det', 7, 'xxx', 'O', '',  True,  True, [0, 35): 'She sold c shills by the Sith Lord.')
                 (6, 6, [25, 29): 'Sith', 'Sith', 'PROPN', 'NNP', 'compound', 7, 'Xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.')
                 (7, 7, [30, 34): 'Lord', 'Lord', 'PROPN', 'NNP', 'pobj', 4, 'Xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.')
                 (8, 8, [34, 35): '.', '.', 'PUNCT', '.', 'punct', 1, '.', 'O', '', False, False, [0, 35): 'She sold c shills by the Sith Lord.')]"""
            ),
        )
        df2 = make_tokens_and_features(
            "She sold c shills by the Sith Lord.",
            _SPACY_LANGUAGE_MODEL,
            add_left_and_right=True,
        )
        # print(f"****{str(df2.to_records())}****")
        self.assertEqual(
            str(df2.to_records()),
            textwrap.dedent(
                """\
                [(0, 0, [0, 3): 'She', '-PRON-', 'PRON', 'PRP', 'nsubj', 1, 'Xxx', 'O', '',  True,  True, [0, 35): 'She sold c shills by the Sith Lord.', <NA>, 1)
                 (1, 1, [4, 8): 'sold', 'sell', 'VERB', 'VBD', 'ROOT', 1, 'xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.', 0, 2)
                 (2, 2, [9, 10): 'c', 'c', 'NOUN', 'NN', 'det', 3, 'x', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.', 1, 3)
                 (3, 3, [11, 17): 'shills', 'shill', 'NOUN', 'NNS', 'dobj', 1, 'xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.', 2, 4)
                 (4, 4, [18, 20): 'by', 'by', 'ADP', 'IN', 'prep', 3, 'xx', 'O', '',  True,  True, [0, 35): 'She sold c shills by the Sith Lord.', 3, 5)
                 (5, 5, [21, 24): 'the', 'the', 'DET', 'DT', 'det', 7, 'xxx', 'O', '',  True,  True, [0, 35): 'She sold c shills by the Sith Lord.', 4, 6)
                 (6, 6, [25, 29): 'Sith', 'Sith', 'PROPN', 'NNP', 'compound', 7, 'Xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.', 5, 7)
                 (7, 7, [30, 34): 'Lord', 'Lord', 'PROPN', 'NNP', 'pobj', 4, 'Xxxx', 'O', '',  True, False, [0, 35): 'She sold c shills by the Sith Lord.', 6, 8)
                 (8, 8, [34, 35): '.', '.', 'PUNCT', '.', 'punct', 1, '.', 'O', '', False, False, [0, 35): 'She sold c shills by the Sith Lord.', 7, <NA>)]"""
            ),
        )

    def test_token_features_to_tree(self):
        df = make_tokens_and_features(
            "Peter Peeper packed a puck of liquid flubber.", _SPACY_LANGUAGE_MODEL
        )
        json = token_features_to_tree(df)
        # print(f"****{json}****")
        expected = {
            "words": [
                {"text": "Peter", "tag": "NNP"},
                {"text": "Peeper", "tag": "NNP"},
                {"text": "packed", "tag": "VBD"},
                {"text": "a", "tag": "DT"},
                {"text": "puck", "tag": "NN"},
                {"text": "of", "tag": "IN"},
                {"text": "liquid", "tag": "JJ"},
                {"text": "flubber", "tag": "NN"},
                {"text": ".", "tag": "."},
            ],
            "arcs": [
                {"start": 0, "end": 1, "label": "compound", "dir": "left"},
                {"start": 1, "end": 2, "label": "nsubj", "dir": "left"},
                {"start": 3, "end": 4, "label": "det", "dir": "left"},
                {"start": 2, "end": 4, "label": "dobj", "dir": "right"},
                {"start": 4, "end": 5, "label": "prep", "dir": "right"},
                {"start": 6, "end": 7, "label": "amod", "dir": "left"},
                {"start": 5, "end": 7, "label": "pobj", "dir": "right"},
                {"start": 2, "end": 8, "label": "punct", "dir": "right"},
            ],
        }
        self.assertDictEqual(json, expected)


if __name__ == "__main__":
    unittest.main()
