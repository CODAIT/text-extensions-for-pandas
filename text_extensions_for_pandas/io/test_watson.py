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

from text_extensions_for_pandas.io.watson import *

'''
text = textwrap.dedent(
  """\
  If Barbie is so popular, why do you have to buy her friends?
  The Bermuda Triangle got tired of warm weather.
  It moved to Alaska. Now Santa Claus is missing.
  -- Steven Wright""")
'''
response = {
  "usage": {
    "text_units": 1,
    "text_characters": 156,
    "features": 4
  },
  "syntax": {
    "tokens": [
      {
        "text": "If",
        "part_of_speech": "SCONJ",
        "location": [
          0,
          2
        ],
        "lemma": "if"
      },
      {
        "text": "Barbie",
        "part_of_speech": "PROPN",
        "location": [
          3,
          9
        ],
        "lemma": "barbie"
      },
      {
        "text": "is",
        "part_of_speech": "AUX",
        "location": [
          10,
          12
        ],
        "lemma": "be"
      },
      {
        "text": "so",
        "part_of_speech": "ADV",
        "location": [
          13,
          15
        ],
        "lemma": "so"
      },
      {
        "text": "popular",
        "part_of_speech": "ADJ",
        "location": [
          16,
          23
        ],
        "lemma": "popular"
      },
      {
        "text": ",",
        "part_of_speech": "PUNCT",
        "location": [
          23,
          24
        ]
      },
      {
        "text": "why",
        "part_of_speech": "ADV",
        "location": [
          25,
          28
        ],
        "lemma": "why"
      },
      {
        "text": "do",
        "part_of_speech": "AUX",
        "location": [
          29,
          31
        ],
        "lemma": "do"
      },
      {
        "text": "you",
        "part_of_speech": "PRON",
        "location": [
          32,
          35
        ],
        "lemma": "you"
      },
      {
        "text": "have",
        "part_of_speech": "VERB",
        "location": [
          36,
          40
        ],
        "lemma": "have"
      },
      {
        "text": "to",
        "part_of_speech": "PART",
        "location": [
          41,
          43
        ],
        "lemma": "to"
      },
      {
        "text": "buy",
        "part_of_speech": "VERB",
        "location": [
          44,
          47
        ],
        "lemma": "buy"
      },
      {
        "text": "her",
        "part_of_speech": "PRON",
        "location": [
          48,
          51
        ],
        "lemma": "her"
      },
      {
        "text": "friends",
        "part_of_speech": "NOUN",
        "location": [
          52,
          59
        ],
        "lemma": "friend"
      },
      {
        "text": "?",
        "part_of_speech": "PUNCT",
        "location": [
          59,
          60
        ]
      },
      {
        "text": "The",
        "part_of_speech": "DET",
        "location": [
          61,
          64
        ],
        "lemma": "the"
      },
      {
        "text": "Bermuda",
        "part_of_speech": "PROPN",
        "location": [
          65,
          72
        ]
      },
      {
        "text": "Triangle",
        "part_of_speech": "PROPN",
        "location": [
          73,
          81
        ],
        "lemma": "triangle"
      },
      {
        "text": "got",
        "part_of_speech": "VERB",
        "location": [
          82,
          85
        ],
        "lemma": "get"
      },
      {
        "text": "tired",
        "part_of_speech": "ADJ",
        "location": [
          86,
          91
        ],
        "lemma": "tired"
      },
      {
        "text": "of",
        "part_of_speech": "ADP",
        "location": [
          92,
          94
        ],
        "lemma": "of"
      },
      {
        "text": "warm",
        "part_of_speech": "ADJ",
        "location": [
          95,
          99
        ],
        "lemma": "warm"
      },
      {
        "text": "weather",
        "part_of_speech": "NOUN",
        "location": [
          100,
          107
        ],
        "lemma": "weather"
      },
      {
        "text": ".",
        "part_of_speech": "PUNCT",
        "location": [
          107,
          108
        ]
      },
      {
        "text": "It",
        "part_of_speech": "PRON",
        "location": [
          109,
          111
        ],
        "lemma": "it"
      },
      {
        "text": "moved",
        "part_of_speech": "VERB",
        "location": [
          112,
          117
        ],
        "lemma": "move"
      },
      {
        "text": "to",
        "part_of_speech": "ADP",
        "location": [
          118,
          120
        ],
        "lemma": "to"
      },
      {
        "text": "Alaska",
        "part_of_speech": "PROPN",
        "location": [
          121,
          127
        ]
      },
      {
        "text": ".",
        "part_of_speech": "PUNCT",
        "location": [
          127,
          128
        ]
      },
      {
        "text": "Now",
        "part_of_speech": "ADV",
        "location": [
          129,
          132
        ],
        "lemma": "now"
      },
      {
        "text": "Santa",
        "part_of_speech": "PROPN",
        "location": [
          133,
          138
        ]
      },
      {
        "text": "Claus",
        "part_of_speech": "PROPN",
        "location": [
          139,
          144
        ]
      },
      {
        "text": "is",
        "part_of_speech": "AUX",
        "location": [
          145,
          147
        ],
        "lemma": "be"
      },
      {
        "text": "missing",
        "part_of_speech": "ADJ",
        "location": [
          148,
          155
        ]
      },
      {
        "text": ".",
        "part_of_speech": "PUNCT",
        "location": [
          155,
          156
        ]
      }
    ],
    "sentences": [
      {
        "text": "If Barbie is so popular, why do you have to buy her friends?",
        "location": [
          0,
          60
        ]
      },
      {
        "text": "The Bermuda Triangle got tired of warm weather.",
        "location": [
          61,
          108
        ]
      },
      {
        "text": "It moved to Alaska.",
        "location": [
          109,
          128
        ]
      },
      {
        "text": "Now Santa Claus is missing.",
        "location": [
          129,
          156
        ]
      }
    ]
  },
  "semantic_roles": [
    {
      "subject": {
        "text": "Barbie"
      },
      "sentence": "If Barbie is so popular, why do you have to buy her friends?",
      "object": {
        "text": "so popular"
      },
      "action": {
        "verb": {
          "text": "be",
          "tense": "present"
        },
        "text": "is",
        "normalized": "be"
      }
    },
    {
      "subject": {
        "text": "you"
      },
      "sentence": "If Barbie is so popular, why do you have to buy her friends?",
      "object": {
        "text": "her friends"
      },
      "action": {
        "verb": {
          "text": "buy",
          "tense": "future"
        },
        "text": "have to buy",
        "normalized": "have to buy"
      }
    },
    {
      "subject": {
        "text": "The Bermuda Triangle"
      },
      "sentence": " The Bermuda Triangle got tired of warm weather.",
      "object": {
        "text": "of warm weather"
      },
      "action": {
        "verb": {
          "text": "tire",
          "tense": "past"
        },
        "text": "tired",
        "normalized": "tire"
      }
    }
  ],
  "relations": [
    {
      "type": "managerOf",
      "sentence": "If Barbie is so popular, why do you have to buy her friends?",
      "score": 0.244055,
      "arguments": [
        {
          "text": "her",
          "location": [
            48,
            51
          ],
          "entities": [
            {
              "type": "Person",
              "text": "her"
            }
          ]
        },
        {
          "text": "friends",
          "location": [
            52,
            59
          ],
          "entities": [
            {
              "type": "Person",
              "text": "friends"
            }
          ]
        }
      ]
    }
  ],
  "language": "en",
  "keywords": [
    {
      "text": "Bermuda Triangle",
      "sentiment": {
        "score": -0.866897,
        "label": "negative"
      },
      "relevance": 0.994976,
      "emotion": {
        "sadness": 0.461508,
        "joy": 0.245015,
        "fear": 0.06578,
        "disgust": 0.044216,
        "anger": 0.248536
      },
      "count": 1
    },
    {
      "text": "Santa Claus",
      "sentiment": {
        "score": -0.940095,
        "label": "negative"
      },
      "relevance": 0.947765,
      "emotion": {
        "sadness": 0.638228,
        "joy": 0.047636,
        "fear": 0.217273,
        "disgust": 0.015876,
        "anger": 0.269789
      },
      "count": 1
    },
    {
      "text": "warm weather",
      "sentiment": {
        "score": -0.866897,
        "label": "negative"
      },
      "relevance": 0.820549,
      "emotion": {
        "sadness": 0.461508,
        "joy": 0.245015,
        "fear": 0.06578,
        "disgust": 0.044216,
        "anger": 0.248536
      },
      "count": 1
    }
  ],
  "entities": [
    {
      "type": "Location",
      "text": "Alaska",
      "sentiment": {
        "score": -0.940095,
        "label": "negative"
      },
      "relevance": 0.978348,
      "count": 1,
      "confidence": 0.999498
    }
  ]
}


class TestWatson(unittest.TestCase):

    def setUp(self):
        # Ensure that diffs are consistent
        pd.set_option("display.max_columns", 250)

    def tearDown(self):
        pd.reset_option("display.max_columns")

    def test_print_all(self):
        result = watson_nlu_parse_response(response)
        print(result['relations'])
        '''for k, v in result.items():
            print("\n\n{}:".format(k))
            print(v)
        '''

    def test_response_entities(self):
        result = watson_nlu_parse_response(response)

        self.assertIn("entities", result)

        expected_types = sorted([e["type"] for e in response["entities"]])
        type_series = result["entities"]["type"]
        self.assertListEqual(sorted(type_series), expected_types)

        expected_text = sorted([e["text"] for e in response["entities"]])
        text_series = result["entities"]["text"]
        self.assertListEqual(sorted(text_series), expected_text)

    def test_response_keywords(self):

        result = watson_nlu_parse_response(response)

        self.assertIn("keywords", result)

        expected_text = sorted([e["text"] for e in response["keywords"]])
        text_series = result["keywords"]["text"]
        self.assertListEqual(sorted(text_series), expected_text)

        expected_relevance = sorted([e["relevance"] for e in response["keywords"]])
        relevance_series = result["keywords"]["relevance"]
        for relevance, expected in zip(expected_relevance, sorted(relevance_series)):
            self.assertAlmostEqual(relevance, expected)

