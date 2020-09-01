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
# span_util.py
#
# Part of text_extensions_for_pandas
#
# Common code that uses classes from both token_span.py and char_span.py.
#

import numpy as np
from typing import *

from text_extensions_for_pandas.array.span import (
    SpanDtype, Span, SpanArray
)
from text_extensions_for_pandas.array.token_span import (
    TokenSpanDtype, TokenSpan, TokenSpanArray
)

_ARRAY_CLASSES = (SpanArray, TokenSpanArray)
_SCALAR_CLASSES = (Span, TokenSpan)
_TOKEN_SPAN_TYPES = (TokenSpan, TokenSpanArray)
_ALL_SPAN_TYPES = (TokenSpan, TokenSpanArray, Span, SpanArray)


def _check_same_tokens(array1, array2):
    if not array1.tokens.equals(array2.tokens):
        raise ValueError(
            f"TokenSpanArrays are over different sets of tokens "
            f"(got {array1.tokens} and {array2.tokens})"
        )


def _check_same_text(o1, o2):
    if not ((o1.target_text is o2.target_text) or (o1.target_text == o2.target_text)):
        raise ValueError(
            f"Spans are over different target text "
            f"(got {o1.target_text} and {o2.target_text})"
        )


def add_spans(span1: Union[TokenSpan, TokenSpanArray, Span, SpanArray],
              span2: Union[TokenSpan, TokenSpanArray, Span, SpanArray])\
        -> Union[TokenSpan, TokenSpanArray, Span, SpanArray]:
    """
    Add a pair of spans and/or span arrays.

    span1 + span2 == minimal span that covers both spans
    :param span1: TokenSpan, Span, TokenSpanArray, or SpanArray
    :param span2: TokenSpan, Span, TokenSpanArray, or SpanArray
    :return: minimal span (or array of spans) that covers both inputs.
    """

    # TODO: Null handling

    if isinstance(span1, TokenSpan) and isinstance(span2, TokenSpan):
        # TokenSpan + TokenSpan = TokenSpan
        _check_same_tokens(span1, span2)
        return TokenSpan(span1.tokens, min(span1.begin_token, span2.begin_token),
                         max(span1.end_token, span2.end_token))
    elif isinstance(span1, Span) and isinstance(span2, Span):
        # Span + *Span = Span
        _check_same_text(span1, span2)
        return Span(span1.target_text, min(span1.begin, span2.begin),
                    max(span1.end, span2.end))
    elif isinstance(span1, _TOKEN_SPAN_TYPES) and isinstance(span2, _TOKEN_SPAN_TYPES):
        # TokenSpanArray + TokenSpan* = TokenSpanArray
        _check_same_tokens(span1, span2)
        return TokenSpanArray(span1.tokens,
                              np.minimum(span1.begin_token, span2.begin_token),
                              np.maximum(span1.end_token, span2.end_token))
    elif (
        (isinstance(span1, SpanArray) and isinstance(span2, _ALL_SPAN_TYPES))
        or (isinstance(span1, _ALL_SPAN_TYPES) and isinstance(span2, SpanArray))
    ):
        # SpanArray + *Span* = SpanArray
        _check_same_text(span1, span2)
        return SpanArray(span1.target_text,
                         np.minimum(span1.begin, span2.begin),
                         np.maximum(span1.end, span2.end))
    else:
        raise TypeError(f"Unexpected combination of span types for add operation: "
                        f"{type(span1)} and {type(span2)}")

