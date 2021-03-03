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
        suffix = html[-800:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """\
</tbody>
</table>
    </div>
    <div id="text"
     style="float:right; border: 1px solid var(--jp-border-color0); width: 60%;">

                <div style="float:center; padding:10px">
                    <p style="font-family:var(--jp-code-font-family); font-size:var(--jp-code-font-size)">
                        <span style="background-color:rgba(255, 215, 0, 0.5)">Item&#39;s</span> <span style="background-color:rgba(255, 215, 0, 0.5)">for</span> <span style="background-color:rgba(255, 215, 0, 0.5)">&lt;</span> <span style="background-color:rgba(255, 215, 0, 0.5)"><span>&#36;</span>100</span> <span style="background-color:rgba(255, 215, 0, 0.5)">&amp;</span> <span style="background-color:rgba(255, 215, 0, 0.5)">change
                    </p>
                </div>

    </div>
</div>
""")

        html = pretty_print_html(_TEST_TOKS["span"].values, False)
        suffix = html[-809:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """\

<div id="text"
 style="float:right; color: var(--jp-layout-color2); border: 1px solid var(--jp-border-color0); width: 100%;">

                <div style="float:center; padding:10px">
                    <p style="font-family:var(--jp-code-font-family); font-size:var(--jp-code-font-size)">
                        <span style="background-color:rgba(255, 215, 0, 0.5)">Item&#39;s</span> <span style="background-color:rgba(255, 215, 0, 0.5)">for</span> <span style="background-color:rgba(255, 215, 0, 0.5)">&lt;</span> <span style="background-color:rgba(255, 215, 0, 0.5)"><span>&#36;</span>100</span> <span style="background-color:rgba(255, 215, 0, 0.5)">&amp;</span> <span style="background-color:rgba(255, 215, 0, 0.5)">change
                    </p>
                </div>

</div>
""")
