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
import regex
import spacy

_SPACY_LANGUAGE_MODEL = spacy.load("en_core_web_sm")

from text_extensions_for_pandas.util import TestBase, pretty_print_html
from text_extensions_for_pandas.io import make_tokens_and_features


_TEST_TEXT = "I will do it if you ask me."
_TEST_TOKS = make_tokens_and_features(_TEST_TEXT, _SPACY_LANGUAGE_MODEL)


class UtilTest(TestBase):
    def test_pretty_print_html(self):
        self.maxDiff = None
        html = pretty_print_html(_TEST_TOKS["token_span"].values)
        suffix = html[-851:]
        # print(f"[[[{suffix}]]]")
        self.assertEqual(
            suffix,
            """    <tr>
      <th>8</th>
      <td>26</td>
      <td>27</td>
      <td>8</td>
      <td>9</td>
      <td>.</td>
    </tr>
  </tbody>
</table>
        </div>
        <div id="text"
         style="float:right; background-color:#F5F5F5; border: 1px solid #E0E0E0; width: 60%;">
            <div style="float:center; padding:10px">
                <p style="font-family:monospace">
                    <span style="background-color:yellow">I</span> <span style="background-color:yellow">will</span> <span style="background-color:yellow">do</span> <span style="background-color:yellow">it</span> <span style="background-color:yellow">if</span> <span style="background-color:yellow">you</span> <span style="background-color:yellow">ask</span> <span style="background-color:yellow">me.
                </p>
            </div>
        </div>
    </div>
    """,
        )

