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

import pandas as pd

# noinspection PyPackageRequirements
import spacy

_SPACY_LANGUAGE_MODEL = spacy.load("en_core_web_sm")

from text_extensions_for_pandas.jupyter import pretty_print_html
from text_extensions_for_pandas.util import TestBase
from text_extensions_for_pandas.io.spacy import make_tokens_and_features


_TEST_TEXT = "Item's for < $100 & change"
_TEST_TOKS = make_tokens_and_features(_TEST_TEXT, _SPACY_LANGUAGE_MODEL)


class JupyterTest(TestBase):
    def test_pretty_print_html(self):
        self.maxDiff = None
        html = pretty_print_html(_TEST_TOKS["span"].values, True)
        suffix = html[-500:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """\
   const Span = window.SpanArray.Span
        const script_context = document.currentScript
        const documents = []

    {
        const doc_spans = Span.arrayFromSpanArray([[0,4],[4,6],[7,10],[11,12],[13,14],[14,17],[18,19],[20,26]])
        const doc_text = 'Item\\'s for < $100 & change'
        documents.push({doc_text: doc_text, doc_spans: doc_spans})
    }

        const instance = new window.SpanArray.SpanArray(documents, true, script_context)
        instance.render()
    }
</script>
""")

        html = pretty_print_html(_TEST_TOKS["span"].values, False)
        suffix = html[-500:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """\
  const Span = window.SpanArray.Span
        const script_context = document.currentScript
        const documents = []

    {
        const doc_spans = Span.arrayFromSpanArray([[0,4],[4,6],[7,10],[11,12],[13,14],[14,17],[18,19],[20,26]])
        const doc_text = 'Item\\'s for < $100 & change'
        documents.push({doc_text: doc_text, doc_spans: doc_spans})
    }

        const instance = new window.SpanArray.SpanArray(documents, false, script_context)
        instance.render()
    }
</script>
""")
