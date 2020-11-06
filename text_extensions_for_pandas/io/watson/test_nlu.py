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
import warnings
import pandas as pd

from text_extensions_for_pandas.io.watson.nlu import *


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

        # The cached data that these tests run off of currently results in many warnings,
        # so we've temporarily disabled warnings for this test suite.
        # TODO: Fix the cached data and re-enable warnings
        warnings.simplefilter("ignore")

    def tearDown(self):
        pd.reset_option("display.max_columns")
        warnings.simplefilter("default")

    @staticmethod
    def load_response_file(filename):
        with open(filename, mode='r') as f:
            return json.load(f)

    def parse_response_file(self, filename):
        response = self.load_response_file(filename)
        return parse_response(response)

    def test_check_basic_response(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        result = self.parse_response_file(filename)
        self.assertEqual(len(result), 6)
        self.assertSequenceEqual(sorted(result.keys()),
                                 ["entities", "entity_mentions",
                                  "keywords", "relations", "semantic_roles", "syntax"])

        # Uncomment to regenerate expected output below.
        # print(f"****{repr(result['syntax'])}****")
        self.assertEqual(
            repr(result["syntax"]),
            # NOTE the escaped backslash in the string below. Be sure to put it back
            # in when regenerating this string!
            textwrap.dedent(
                """\
                                     span part_of_speech     lemma  \\
                0            [0, 2): 'If'          SCONJ        if   
                1        [3, 9): 'Barbie'          PROPN    barbie   
                2          [10, 12): 'is'            AUX        be   
                3          [13, 15): 'so'            ADV        so   
                4     [16, 23): 'popular'            ADJ   popular   
                5           [23, 24): ','          PUNCT      None   
                6         [25, 28): 'why'            ADV       why   
                7          [29, 31): 'do'            AUX        do   
                8         [32, 35): 'you'           PRON       you   
                9        [36, 40): 'have'           VERB      have   
                10         [41, 43): 'to'           PART        to   
                11        [44, 47): 'buy'           VERB       buy   
                12        [48, 51): 'her'           PRON       her   
                13    [52, 59): 'friends'           NOUN    friend   
                14          [59, 60): '?'          PUNCT      None   
                15        [61, 64): 'The'            DET       the   
                16    [65, 72): 'Bermuda'          PROPN      None   
                17   [73, 81): 'Triangle'          PROPN  triangle   
                18        [82, 85): 'got'           VERB       get   
                19      [86, 91): 'tired'            ADJ     tired   
                20         [92, 94): 'of'            ADP        of   
                21       [95, 99): 'warm'            ADJ      warm   
                22  [100, 107): 'weather'           NOUN   weather   
                23        [107, 108): '.'          PUNCT      None   
                24       [109, 111): 'It'           PRON        it   
                25    [112, 117): 'moved'           VERB      move   
                26       [118, 120): 'to'            ADP        to   
                27   [121, 127): 'Alaska'          PROPN      None   
                28        [127, 128): '.'          PUNCT      None   
                29      [129, 132): 'Now'            ADV       now   
                30    [133, 138): 'Santa'          PROPN      None   
                31    [139, 144): 'Claus'          PROPN      None   
                32       [145, 147): 'is'            AUX        be   
                33  [148, 155): 'missing'            ADJ      None   
                34        [155, 156): '.'          PUNCT      None   
                
                                                             sentence  
                0   [0, 60): 'If Barbie is so popular, why do you ...  
                1   [0, 60): 'If Barbie is so popular, why do you ...  
                2   [0, 60): 'If Barbie is so popular, why do you ...  
                3   [0, 60): 'If Barbie is so popular, why do you ...  
                4   [0, 60): 'If Barbie is so popular, why do you ...  
                5   [0, 60): 'If Barbie is so popular, why do you ...  
                6   [0, 60): 'If Barbie is so popular, why do you ...  
                7   [0, 60): 'If Barbie is so popular, why do you ...  
                8   [0, 60): 'If Barbie is so popular, why do you ...  
                9   [0, 60): 'If Barbie is so popular, why do you ...  
                10  [0, 60): 'If Barbie is so popular, why do you ...  
                11  [0, 60): 'If Barbie is so popular, why do you ...  
                12  [0, 60): 'If Barbie is so popular, why do you ...  
                13  [0, 60): 'If Barbie is so popular, why do you ...  
                14  [0, 60): 'If Barbie is so popular, why do you ...  
                15  [61, 108): 'The Bermuda Triangle got tired of ...  
                16  [61, 108): 'The Bermuda Triangle got tired of ...  
                17  [61, 108): 'The Bermuda Triangle got tired of ...  
                18  [61, 108): 'The Bermuda Triangle got tired of ...  
                19  [61, 108): 'The Bermuda Triangle got tired of ...  
                20  [61, 108): 'The Bermuda Triangle got tired of ...  
                21  [61, 108): 'The Bermuda Triangle got tired of ...  
                22  [61, 108): 'The Bermuda Triangle got tired of ...  
                23  [61, 108): 'The Bermuda Triangle got tired of ...  
                24                  [109, 128): 'It moved to Alaska.'  
                25                  [109, 128): 'It moved to Alaska.'  
                26                  [109, 128): 'It moved to Alaska.'  
                27                  [109, 128): 'It moved to Alaska.'  
                28                  [109, 128): 'It moved to Alaska.'  
                29          [129, 156): 'Now Santa Claus is missing.'  
                30          [129, 156): 'Now Santa Claus is missing.'  
                31          [129, 156): 'Now Santa Claus is missing.'  
                32          [129, 156): 'Now Santa Claus is missing.'  
                33          [129, 156): 'Now Santa Claus is missing.'  
                34          [129, 156): 'Now Santa Claus is missing.'  """
            ),
        )

        self.assertEqual(
            repr(result["entities"]),
            textwrap.dedent(
                """\
                       type    text sentiment.label  sentiment.score  relevance  count  \\
                0  Location  Alaska        negative        -0.940095   0.978348      1   
                
                   confidence  
                0    0.999498  """
            ),
        )

        self.assertEqual(
            repr(result["keywords"]),
            textwrap.dedent(
                """\
                               text sentiment.label  sentiment.score  relevance  \\
                0  Bermuda Triangle        negative        -0.866897   0.994976   
                1       Santa Claus        negative        -0.940095   0.947765   
                2      warm weather        negative        -0.866897   0.820549   
                
                   emotion.sadness  emotion.joy  emotion.fear  emotion.disgust  emotion.anger  \\
                0         0.461508     0.245015      0.065780         0.044216       0.248536   
                1         0.638228     0.047636      0.217273         0.015876       0.269789   
                2         0.461508     0.245015      0.065780         0.044216       0.248536   
                
                   count  
                0      1  
                1      1  
                2      1  """
            ),
        )

        self.assertEqual(
            repr(result["relations"]),
            textwrap.dedent(
                """\
                        type                                      sentence_span     score  \\
                0  managerOf  [0, 60): 'If Barbie is so popular, why do you ...  0.244055   
                
                  arguments.0.span     arguments.1.span arguments.0.entities.type  \\
                0  [48, 51): 'her'  [52, 59): 'friends'                    Person   
                
                  arguments.1.entities.type arguments.0.entities.text  \\
                0                    Person                       her   
                
                  arguments.1.entities.text  
                0                   friends  """
            ),
        )

        self.assertEqual(
            repr(result["semantic_roles"]),
            textwrap.dedent(
                """\
                           subject.text                                           sentence  \\
                0                Barbie  If Barbie is so popular, why do you have to bu...   
                1                   you  If Barbie is so popular, why do you have to bu...   
                2  The Bermuda Triangle    The Bermuda Triangle got tired of warm weather.   
                
                       object.text action.verb.text action.verb.tense  action.text  \\
                0       so popular               be           present           is   
                1      her friends              buy            future  have to buy   
                2  of warm weather             tire              past        tired   
                
                  action.normalized  
                0                be  
                1       have to buy  
                2              tire  """
            ),
        )

    def test_large_response(self):
        filename = "test_data/io/test_watson/holy_grail_response.txt"
        result = self.parse_response_file(filename)
        self.assertEqual(len(result), 6)
        self.assertSequenceEqual(sorted(result.keys()),
                                 ["entities", "entity_mentions",
                                  "keywords", "relations", "semantic_roles",
                                  "syntax"])

    def test_entity_mentions(self):
        filename = "test_data/io/test_watson/mentions_response.txt"
        # Make sure the file loads.
        response = self.load_response_file(filename)
        result = parse_response(response)
        self.assertEqual(
            repr(result["entities"]),
            textwrap.dedent("""\
               type           text sentiment.label  sentiment.score  relevance  count  \\
        0    Person  Steven Wright        negative        -0.866897   0.978348      2   
        1  Location         Alaska        negative        -0.940095   0.277941      1   
        
           confidence  
        0    0.999997  
        1    0.999498  """)
        )
        self.assertEqual(
            repr(result["entity_mentions"]),
            textwrap.dedent("""\
                       type           text                         span  confidence
                0    Person  Steven Wright    [64, 77): 'Steven Wright'    0.998251
                1    Person  Steven Wright  [177, 190): 'Steven Wright'    0.998251
                2  Location         Alaska         [138, 144): 'Alaska'    0.999498""")
        )

    def test_response_entities(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = parse_response(response)

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

        self.assertIn("entities", dfs)
        self.assertIn("text", dfs["entities"].columns)
        char_span = dfs["syntax"]["span"].values

        token_span = make_span_from_entities(char_span, dfs["entities"])
        self.assertEqual(len(token_span), 24)
        self.assertEqual(token_span[0].covered_text, "Monty Python")

    def test_response_keywords(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = parse_response(response)

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
        result = parse_response(response)

        self.assertIn("relations", result)
        df = result["relations"]

        self.assertIn("arguments.0.span", df.columns)
        self.assertIn("arguments.1.span", df.columns)
        self.assertIn("sentence_span", df.columns)

    def test_response_syntax(self):
        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = parse_response(response)

        self.assertIn("syntax", result)
        df = result["syntax"]

        self.assertIn("span", df.columns)
        self.assertIn("sentence", df.columns)

    def test_empty_frames(self):
        empty_response = {
            "usage": {
                "text_units": 1,
                "text_characters": 156,
                "features": 4
            },
            "language": "en",
        }

        with self.assertWarns(UserWarning) as cm:
            empty_result = parse_response(empty_response, apply_standard_schema=True)

            expected_warn = "Did not receive and could not build original text"
            warn = cm.warnings[0]
            self.assertEqual(expected_warn, str(warn.message))

        for df in empty_result.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 0)

        filename = "test_data/io/test_watson/basic_response.txt"
        response = self.load_response_file(filename)
        result = parse_response(response, apply_standard_schema=True)

        for key, df in result.items():
            self.assertIsInstance(df, pd.DataFrame)
            if key != "entity_mentions":  # No entity mentions in test response
                self.assertGreater(len(df), 0)

        for name, empty_df in empty_result.items():
            empty_response_cols = list(empty_df.columns)
            response_cols = list(result[name].columns)
            self.assertListEqual(response_cols, empty_response_cols)


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

    def test_expected_relations_response(self):
        relations = self.response["relations"]
        r = relations[0]

        # Check expected columns
        self.assertIn("type", r)
        self.assertIn("sentence", r)
        self.assertIn("score", r)
        self.assertIn("arguments", r)

        # Check expected format of arguments
        args = r["arguments"]
        self.assertEqual(len(args), 2)
        arg1 = args[0]
        self.assertIn("text", arg1)
        self.assertIn("location", arg1)

        # Check location format
        loc = arg1["location"]
        self.assertIsInstance(loc, list)
        self.assertEqual(len(loc), 2)
        self.assertTrue(all([isinstance(i, int) for i in loc]))

    def test_expected_syntax_response(self):
        syntax_response = self.response["syntax"]
        tokens = syntax_response["tokens"]

        # Check tokens format
        r = tokens[0]
        self.assertIn("text", r)
        self.assertIn("location", r)
        loc = r["location"]
        self.assertIsInstance(loc, list)
        self.assertEqual(len(loc), 2)
        self.assertTrue(all([isinstance(i, int) for i in loc]))

        # Check sentence format
        sentence = syntax_response["sentences"]
        r = sentence[0]
        self.assertIn("text", r)
        self.assertIn("location", r)
        loc = r["location"]
        self.assertIsInstance(loc, list)
        self.assertEqual(len(loc), 2)
        self.assertTrue(all([isinstance(i, int) for i in loc]))
