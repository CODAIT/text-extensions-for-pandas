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

#
# project.py
#
# Projection functions (functions that take values from one tuple and return a
# scalar) for spans.
#

import pandas as pd

from typing import *

# Internal imports
from text_extensions_for_pandas.array.span import (
    SpanArray, Span
)
from text_extensions_for_pandas.array.token_span import (
    TokenSpanArray
)


def lemmatize(spans: Union[pd.Series, SpanArray, Iterable[Span]],
              token_features: pd.DataFrame,
              lemma_col_name: str = "lemma",
              token_span_col_name: str = "span") -> List[str]:
    """
    Convert spans to their normal form using lemma information in a token
    features table.

    :param spans: Spans to be normalized. Each may represent zero or more
    tokens.

    :param token_features: DataFrame of token metadata. Index must be aligned
    with the token indices in `spans`.

    :param lemma_col_name: Optional custom name for the DataFrame column
    containing the lemmatized form of each token.

    :param token_span_col_name: Optional custom name for the DataFrame column
    containing the span of each token.

    :return: A list containing normalized versions of the tokens
    in `spans`, with each token separated by single space character.
    """
    char_spans = SpanArray.make_array(spans)
    token_spans = TokenSpanArray.align_to_tokens(token_features[token_span_col_name],
                                                 char_spans)
    ret = []  # Type: List[str]
    # TODO: Vectorize this loop
    for i in range(len(token_spans)):
        lemmas = token_features[lemma_col_name][
                 token_spans.begin_token[i]:token_spans.end_token[i]
                 ]
        ret.append(" ".join(lemmas))
    return ret


