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
import os
import textwrap
import unittest

from text_extensions_for_pandas.io.watson import *


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

    def test_make_span_from_entities(self):
        filename = "test_data/io/test_watson/holy_grail_response.txt"
        dfs = self.parse_response_file(filename)

        self.assertIn('entities', dfs)
        self.assertIn('text', dfs['entities'].columns)
        char_span = dfs['syntax']['char_span'].values

        token_span = make_span_from_entities(char_span, dfs['entities'])
        x = 1
        # TODO check span

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

        self.assertIn("arguments.0.span", df.columns)
        self.assertIn("arguments.1.span", df.columns)
        self.assertIn("sentence_span", df.columns)

    def test_response_syntax(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = watson_nlu_parse_response(response)

        self.assertIn("syntax", result)
        df = result["syntax"]

        self.assertIn("char_span", df.columns)
        self.assertIn("token_span", df.columns)
        self.assertIn("sentence", df.columns)


@unittest.skipIf(os.environ.get("IBM_API_KEY") is None, "Env var 'IBM_API_KEY' is not set")
class TestWatsonApiHandling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.response = TestWatsonApiHandling._make_request()

    @staticmethod
    def _make_request():
        from ibm_watson import NaturalLanguageUnderstandingV1
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
        from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, ConceptsOptions, EmotionOptions, EntitiesOptions, KeywordsOptions, \
            MetadataOptions, RelationsOptions, SemanticRolesOptions, SentimentOptions, SyntaxOptions, SyntaxOptionsTokens

        # Retrieve the APIKEY for authentication
        apikey = os.environ.get("IBM_API_KEY")
        if apikey is None:
            raise ValueError("Expected apikey in the environment variable 'IBM_API_KEY'")

        # Get the service URL for your IBM Cloud instance
        ibm_cloud_service_url = os.environ.get("IBM_SERVICE_URL")
        if ibm_cloud_service_url is None:
            raise ValueError("Expected IBM cloud service URL in the environment variable 'IBM_SERVICE_URL'")

        # Initialize the authenticator for making requests
        authenticator = IAMAuthenticator(apikey)
        natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2019-07-12',
            authenticator=authenticator
        )

        natural_language_understanding.set_service_url(ibm_cloud_service_url)

        response = natural_language_understanding.analyze(
            url="https://raw.githubusercontent.com/CODAIT/text-extensions-for-pandas/master/resources/holy_grail.txt",
            return_analyzed_text=True,
            features=Features(
                #categories=CategoriesOptions(limit=3),
                #concepts=ConceptsOptions(limit=3),
                #emotion=EmotionOptions(targets=['grail']),
                entities=EntitiesOptions(sentiment=True),
                keywords=KeywordsOptions(sentiment=True,emotion=True),
                #metadata=MetadataOptions(),
                relations=RelationsOptions(),
                semantic_roles=SemanticRolesOptions(),
                #sentiment=SentimentOptions(targets=['Arthur']),
                syntax=SyntaxOptions(sentences=True, tokens=SyntaxOptionsTokens(lemma=True, part_of_speech=True))  # Experimental
            )).get_result()

        return response

    def test_expected_features(self):
        self.assertIn("entities", self.response)
        self.assertIsInstance(self.response["entities"], list)
        self.assertIn("keywords", self.response)
        self.assertIsInstance(self.response["keywords"], list)
        self.assertIn("relations", self.response)
        self.assertIsInstance(self.response["relations"], list)
        self.assertIn("semantic_roles", self.response)
        self.assertIsInstance(self.response["semantic_roles"], list)
        self.assertIn("syntax", self.response)
        syntax = self.response["syntax"]
        self.assertIsInstance(syntax, dict)
        self.assertIn("tokens", syntax)
        self.assertIn("sentences", syntax)

    def test_analyzed_text_present(self):
        self.assertIn("analyzed_text", self.response)
