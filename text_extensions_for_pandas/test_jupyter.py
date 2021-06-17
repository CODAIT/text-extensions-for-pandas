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
        suffix = html[-380:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """\
ndow.SpanArray.render = render
}
        const Entry = window.SpanArray.Entry
        const render = window.SpanArray.render
        const spanArray = [[0,4],[4,6],[7,10],[11,12],[13,14],[14,17],[18,19],[20,26]]
        const entries = Entry.fromSpanArray(spanArray)
        const doc_text = `Item's for < $100 & change`
        render(doc_text, entries, 1, true)
    }
</script>
""")

        html = pretty_print_html(_TEST_TOKS["span"].values, False)
        suffix = html[-380:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """\
dow.SpanArray.render = render
}
        const Entry = window.SpanArray.Entry
        const render = window.SpanArray.render
        const spanArray = [[0,4],[4,6],[7,10],[11,12],[13,14],[14,17],[18,19],[20,26]]
        const entries = Entry.fromSpanArray(spanArray)
        const doc_text = `Item's for < $100 & change`
        render(doc_text, entries, 2, false)
    }
</script>
""")
