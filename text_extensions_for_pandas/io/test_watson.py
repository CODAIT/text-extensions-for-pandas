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
  Who is General Failure and why is he reading my hard disk?
  If Barbie is so popular, why do you have to buy her friends?""")
'''
response1 = {
  "usage": {
    "text_units": 1,
    "text_characters": 119,
    "features": 4
  },
  "syntax": {
    "tokens": [
      {
        "text": "Who",
        "part_of_speech": "PRON",
        "location": [
          0,
          3
        ],
        "lemma": "who"
      },
      {
        "text": "is",
        "part_of_speech": "AUX",
        "location": [
          4,
          6
        ],
        "lemma": "be"
      },
      {
        "text": "General",
        "part_of_speech": "ADJ",
        "location": [
          7,
          14
        ],
        "lemma": "general"
      },
      {
        "text": "Failure",
        "part_of_speech": "NOUN",
        "location": [
          15,
          22
        ],
        "lemma": "failure"
      },
      {
        "text": "and",
        "part_of_speech": "CCONJ",
        "location": [
          23,
          26
        ],
        "lemma": "and"
      },
      {
        "text": "why",
        "part_of_speech": "ADV",
        "location": [
          27,
          30
        ],
        "lemma": "why"
      },
      {
        "text": "is",
        "part_of_speech": "AUX",
        "location": [
          31,
          33
        ],
        "lemma": "be"
      },
      {
        "text": "he",
        "part_of_speech": "PRON",
        "location": [
          34,
          36
        ],
        "lemma": "he"
      },
      {
        "text": "reading",
        "part_of_speech": "VERB",
        "location": [
          37,
          44
        ],
        "lemma": "read"
      },
      {
        "text": "my",
        "part_of_speech": "PRON",
        "location": [
          45,
          47
        ],
        "lemma": "my"
      },
      {
        "text": "hard",
        "part_of_speech": "ADJ",
        "location": [
          48,
          52
        ],
        "lemma": "hard"
      },
      {
        "text": "disk",
        "part_of_speech": "NOUN",
        "location": [
          53,
          57
        ],
        "lemma": "disc"
      },
      {
        "text": "?",
        "part_of_speech": "PUNCT",
        "location": [
          57,
          58
        ]
      },
      {
        "text": "If",
        "part_of_speech": "SCONJ",
        "location": [
          59,
          61
        ],
        "lemma": "if"
      },
      {
        "text": "Barbie",
        "part_of_speech": "PROPN",
        "location": [
          62,
          68
        ],
        "lemma": "barbie"
      },
      {
        "text": "is",
        "part_of_speech": "AUX",
        "location": [
          69,
          71
        ],
        "lemma": "be"
      },
      {
        "text": "so",
        "part_of_speech": "ADV",
        "location": [
          72,
          74
        ],
        "lemma": "so"
      },
      {
        "text": "popular",
        "part_of_speech": "ADJ",
        "location": [
          75,
          82
        ],
        "lemma": "popular"
      },
      {
        "text": ",",
        "part_of_speech": "PUNCT",
        "location": [
          82,
          83
        ]
      },
      {
        "text": "why",
        "part_of_speech": "ADV",
        "location": [
          84,
          87
        ],
        "lemma": "why"
      },
      {
        "text": "do",
        "part_of_speech": "AUX",
        "location": [
          88,
          90
        ],
        "lemma": "do"
      },
      {
        "text": "you",
        "part_of_speech": "PRON",
        "location": [
          91,
          94
        ],
        "lemma": "you"
      },
      {
        "text": "have",
        "part_of_speech": "VERB",
        "location": [
          95,
          99
        ],
        "lemma": "have"
      },
      {
        "text": "to",
        "part_of_speech": "PART",
        "location": [
          100,
          102
        ],
        "lemma": "to"
      },
      {
        "text": "buy",
        "part_of_speech": "VERB",
        "location": [
          103,
          106
        ],
        "lemma": "buy"
      },
      {
        "text": "her",
        "part_of_speech": "PRON",
        "location": [
          107,
          110
        ],
        "lemma": "her"
      },
      {
        "text": "friends",
        "part_of_speech": "NOUN",
        "location": [
          111,
          118
        ],
        "lemma": "friend"
      },
      {
        "text": "?",
        "part_of_speech": "PUNCT",
        "location": [
          118,
          119
        ]
      }
    ],
    "sentences": [
      {
        "text": "Who is General Failure and why is he reading my hard disk?",
        "location": [
          0,
          58
        ]
      },
      {
        "text": "If Barbie is so popular, why do you have to buy her friends?",
        "location": [
          59,
          119
        ]
      }
    ]
  },
  "semantic_roles": [
    {
      "subject": {
        "text": "he"
      },
      "sentence": "Who is General Failure and why is he reading my hard disk?",
      "object": {
        "text": "reading my hard disk"
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
        "text": "he"
      },
      "sentence": "Who is General Failure and why is he reading my hard disk?",
      "object": {
        "text": "my hard disk"
      },
      "action": {
        "verb": {
          "text": "read",
          "tense": "present"
        },
        "text": "reading",
        "normalized": "read"
      }
    },
    {
      "subject": {
        "text": "Barbie"
      },
      "sentence": " If Barbie is so popular, why do you have to buy her friends?",
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
    }
  ],
  "relations": [
    {
      "type": "employedBy",
      "sentence": "Who is General Failure and why is he reading my hard disk?",
      "score": 0.547749,
      "arguments": [
        {
          "text": "Who",
          "location": [
            0,
            3
          ],
          "entities": [
            {
              "type": "Person",
              "text": "Who"
            }
          ]
        },
        {
          "text": "General Failure",
          "location": [
            7,
            22
          ],
          "entities": [
            {
              "type": "Organization",
              "text": "General Failure",
              "disambiguation": {
                "subtype": [
                  "Government"
                ]
              }
            }
          ]
        }
      ]
    },
    {
      "type": "managerOf",
      "sentence": "If Barbie is so popular, why do you have to buy her friends?",
      "score": 0.244055,
      "arguments": [
        {
          "text": "her",
          "location": [
            107,
            110
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
            111,
            118
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
      "text": "General Failure",
      "sentiment": {
        "score": -0.830962,
        "label": "negative"
      },
      "relevance": 0.985749,
      "emotion": {
        "sadness": 0.695442,
        "joy": 0.017928,
        "fear": 0.290046,
        "disgust": 0.177563,
        "anger": 0.05669
      },
      "count": 1
    },
    {
      "text": "hard disk",
      "sentiment": {
        "score": -0.830962,
        "label": "negative"
      },
      "relevance": 0.911891,
      "emotion": {
        "sadness": 0.695442,
        "joy": 0.017928,
        "fear": 0.290046,
        "disgust": 0.177563,
        "anger": 0.05669
      },
      "count": 1
    },
    {
      "text": "Barbie",
      "sentiment": {
        "score": 0,
        "label": "neutral"
      },
      "relevance": 0.772045,
      "emotion": {
        "sadness": 0.17992,
        "joy": 0.11034,
        "fear": 0.06123,
        "disgust": 0.147195,
        "anger": 0.09765
      },
      "count": 1
    }
  ],
  "entities": []
}

'''
text = textwrap.dedent(
  """\
  The Bermuda Triangle got tired of warm weather.
  It moved to Alaska. Now Santa Claus is missing.
  -- Steven Wright""")
'''
response2 = {
  "usage": {
    "text_units": 1,
    "text_characters": 113,
    "features": 4
  },
  "syntax": {
    "tokens": [
      {
        "text": "The",
        "part_of_speech": "DET",
        "location": [
          0,
          3
        ],
        "lemma": "the"
      },
      {
        "text": "Bermuda",
        "part_of_speech": "PROPN",
        "location": [
          4,
          11
        ]
      },
      {
        "text": "Triangle",
        "part_of_speech": "PROPN",
        "location": [
          12,
          20
        ],
        "lemma": "triangle"
      },
      {
        "text": "got",
        "part_of_speech": "VERB",
        "location": [
          21,
          24
        ],
        "lemma": "get"
      },
      {
        "text": "tired",
        "part_of_speech": "ADJ",
        "location": [
          25,
          30
        ],
        "lemma": "tired"
      },
      {
        "text": "of",
        "part_of_speech": "ADP",
        "location": [
          31,
          33
        ],
        "lemma": "of"
      },
      {
        "text": "warm",
        "part_of_speech": "ADJ",
        "location": [
          34,
          38
        ],
        "lemma": "warm"
      },
      {
        "text": "weather",
        "part_of_speech": "NOUN",
        "location": [
          39,
          46
        ],
        "lemma": "weather"
      },
      {
        "text": ".",
        "part_of_speech": "PUNCT",
        "location": [
          46,
          47
        ]
      },
      {
        "text": "It",
        "part_of_speech": "PRON",
        "location": [
          49,
          51
        ],
        "lemma": "it"
      },
      {
        "text": "moved",
        "part_of_speech": "VERB",
        "location": [
          52,
          57
        ],
        "lemma": "move"
      },
      {
        "text": "to",
        "part_of_speech": "ADP",
        "location": [
          58,
          60
        ],
        "lemma": "to"
      },
      {
        "text": "Alaska",
        "part_of_speech": "PROPN",
        "location": [
          61,
          67
        ]
      },
      {
        "text": ".",
        "part_of_speech": "PUNCT",
        "location": [
          67,
          68
        ]
      },
      {
        "text": "Now",
        "part_of_speech": "ADV",
        "location": [
          69,
          72
        ],
        "lemma": "now"
      },
      {
        "text": "Santa",
        "part_of_speech": "PROPN",
        "location": [
          73,
          78
        ]
      },
      {
        "text": "Claus",
        "part_of_speech": "PROPN",
        "location": [
          79,
          84
        ]
      },
      {
        "text": "is",
        "part_of_speech": "AUX",
        "location": [
          85,
          87
        ],
        "lemma": "be"
      },
      {
        "text": "missing",
        "part_of_speech": "ADJ",
        "location": [
          88,
          95
        ]
      },
      {
        "text": ".",
        "part_of_speech": "PUNCT",
        "location": [
          95,
          96
        ]
      },
      {
        "text": "--",
        "part_of_speech": "PUNCT",
        "location": [
          97,
          99
        ]
      },
      {
        "text": "Steven",
        "part_of_speech": "PROPN",
        "location": [
          100,
          106
        ]
      },
      {
        "text": "Wright",
        "part_of_speech": "PROPN",
        "location": [
          107,
          113
        ]
      }
    ],
    "sentences": [
      {
        "text": "The Bermuda Triangle got tired of warm weather.",
        "location": [
          0,
          47
        ]
      },
      {
        "text": "It moved to Alaska.",
        "location": [
          49,
          68
        ]
      },
      {
        "text": "Now Santa Claus is missing.",
        "location": [
          69,
          96
        ]
      },
      {
        "text": "-- Steven Wright",
        "location": [
          97,
          113
        ]
      }
    ]
  },
  "semantic_roles": [
    {
      "subject": {
        "text": "The Bermuda Triangle"
      },
      "sentence": "The Bermuda Triangle got tired of warm weather.",
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
    },
    {
      "subject": {
        "text": "Santa Claus"
      },
      "sentence": " Now Santa Claus is missing.",
      "action": {
        "verb": {
          "text": "miss",
          "tense": "present"
        },
        "text": "is missing",
        "normalized": "be miss"
      }
    }
  ],
  "relations": [],
  "language": "en",
  "keywords": [
    {
      "text": "Bermuda Triangle",
      "sentiment": {
        "score": -0.866897,
        "label": "negative"
      },
      "relevance": 0.998991,
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
      "relevance": 0.977698,
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
      "relevance": 0.878971,
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
      "relevance": 0.963296,
      "count": 1,
      "confidence": 0.999498
    },
    {
      "type": "Person",
      "text": "Steven Wright",
      "sentiment": {
        "score": 0,
        "label": "neutral"
      },
      "relevance": 0.252317,
      "count": 1,
      "confidence": 0.998251
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
        result = watson_nlu_parse_response(response1)
        for k, v in result.items():
            print("\n\n{}:".format(k))
            print(v)

    def test_response_entities(self):
        result = watson_nlu_parse_response(response2)

        self.assertIn("entities", result)

        expected_types = sorted([e["type"] for e in response2["entities"]])
        type_series = result["entities"]["type"]
        self.assertListEqual(sorted(type_series), expected_types)

        expected_text = sorted([e["text"] for e in response2["entities"]])
        text_series = result["entities"]["text"]
        self.assertListEqual(sorted(text_series), expected_text)

    def test_response_keywords(self):

        result = watson_nlu_parse_response(response2)

        self.assertIn("keywords", result)

        expected_text = sorted([e["text"] for e in response2["keywords"]])
        text_series = result["keywords"]["text"]
        self.assertListEqual(sorted(text_series), expected_text)

        expected_relevance = sorted([e["relevance"] for e in response2["keywords"]])
        relevance_series = result["keywords"]["relevance"]
        for relevance, expected in zip(expected_relevance, sorted(relevance_series)):
            self.assertAlmostEqual(relevance, expected)

