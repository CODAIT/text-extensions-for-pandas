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

import json
import textwrap
import unittest

from text_extensions_for_pandas.io.watson import *

'''

'''


class TestWatson(unittest.TestCase):

    def setUp(self):
        # Ensure that diffs are consistent
        pd.set_option("display.max_columns", 250)

        # Text for basic_response.txt
        self.basic_text = textwrap.dedent(
          """\
          If Barbie is so popular, why do you have to buy her friends?
          The Bermuda Triangle got tired of warm weather.
          It moved to Alaska. Now Santa Claus is missing.
          -- Steven Wright""")

    def tearDown(self):
        pd.reset_option("display.max_columns")

    def load_response_file(self, filename):
        with open(filename, mode='r') as f:
            return json.load(f)

    def parse_response_file(self, filename):
        response = self.load_response_file(filename)
        return watson_nlu_parse_response(response)

    def test_print_reponse(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        result = self.parse_response_file(filename)
        for k, v in result.items():
            print("\n\n{}:".format(k))
            print(v)

    def test_large_response(self):
        filename = "test_data/io/test_watson/holy_grail_response.txt"
        result = self.parse_response_file(filename)
        print(result['relations'])

    def test_response_entities(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = watson_nlu_parse_response(response)

        self.assertIn("entities", result)

        expected_types = sorted([e["type"] for e in response["entities"]])
        type_series = result["entities"]["type"]
        self.assertListEqual(sorted(type_series), expected_types)

        expected_text = sorted([e["text"] for e in response["entities"]])
        text_series = result["entities"]["text"]
        self.assertListEqual(sorted(text_series), expected_text)

    def test_response_keywords(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = watson_nlu_parse_response(response)

        self.assertIn("keywords", result)

        expected_text = sorted([e["text"] for e in response["keywords"]])
        text_series = result["keywords"]["text"]
        self.assertListEqual(sorted(text_series), expected_text)

        expected_relevance = sorted([e["relevance"] for e in response["keywords"]])
        relevance_series = result["keywords"]["relevance"]
        for relevance, expected in zip(expected_relevance, sorted(relevance_series)):
            self.assertAlmostEqual(relevance, expected)

    def test_response_relations(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = watson_nlu_parse_response(response)

        self.assertIn("relations", result)
        df = result["relations"]

        self.assertIn("arguments.0.text", df.columns)
        self.assertIn("arguments.1.text", df.columns)
        self.assertIn("arguments.0.location", df.columns)
        self.assertIn("arguments.1.location", df.columns)

    def test_response_relations(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = watson_nlu_parse_response(response)

        self.assertIn("syntax", result)
        df = result["syntax"]

        self.assertIn("char_span", df.columns)
        self.assertIn("token_span", df.columns)
        self.assertIn("sentence", df.columns)
