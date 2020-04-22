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

################################################################################
# tokenize.py
#
# Functions for tokenization of text

import numpy as np
import pandas as pd

from text_extensions_for_pandas.array import (
    CharSpanArray,
    TokenSpanArray,
)


def make_bert_tokens(target_text: str, tokenizer) -> pd.DataFrame:
    """
    Tokenize the indicated text for BERT embeddings and return a DataFrame
    with one row per token.

    :param: target_text: string to tokenize
    :param: tokenizer: A tokenizer that is a subclass of huggingface transformers
                       PreTrainingTokenizerFast which supports `encode_plus` with
                       return_offsets_mapping=True.

    :returns: A `pd.DataFrame` with the following columns:
     * "id": unique integer ID for each token
     * "char_span": span of the token with character offsets
     * "token_span": span of the token with token offsets
     * "input_id": integer ID suitable for input to a BERT embedding model
     * "token_type_id": list of token type ids to be fed to a model
     * "attention_mask": list of indices specifying which tokens should be
                         attended to by the model
     * "special_tokens_mask": `True` if the token is a zero-length special token
       such as "start of document"
    """
    from transformers.tokenization_utils import PreTrainedTokenizerFast
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Tokenizer must be an instance of "
                        "transformers.PreTrainedTokenizerFast that supports "
                        "encode_plus with return_offsets_mapping=True.")
    tokenized_result = tokenizer.encode_plus(target_text,
                                             return_special_tokens_mask=True,
                                             return_offsets_mapping=True)

    # Get offset mapping from tokenizer
    offsets = tokenized_result["offset_mapping"]

    # Init any special tokens at beginning
    i = 0
    while offsets[i] is None:
        offsets[i] = (0, 0)
        i += 1

    # Make a DataFrame to unzip (begin, end) offsets
    offset_df = pd.DataFrame(offsets, columns=["begin", "end"])

    # Convert special tokens mask to boolean
    special_tokens_mask = \
        pd.Series(tokenized_result["special_tokens_mask"]).astype('bool')

    # Fill remaining special tokens to zero-length spans
    ends = offset_df['end'].fillna(method='ffill').astype('int32')
    begins = offset_df['begin'].mask(special_tokens_mask, other=ends).astype('int32')

    # Create char and token span arrays
    char_spans = CharSpanArray(target_text, begins, ends)
    token_spans = TokenSpanArray(char_spans,
                                 np.arange(len(char_spans)),
                                 np.arange(1, len(char_spans) + 1))

    token_features = pd.DataFrame({
        "id": special_tokens_mask.index,
        # Use values instead of series because different indexes
        "char_span": pd.Series(char_spans).values,
        "token_span": pd.Series(token_spans).values,
        "input_id": tokenized_result["input_ids"],
        "token_type_id": tokenized_result["token_type_ids"],
        "attention_mask": tokenized_result["attention_mask"],
        "special_tokens_mask": special_tokens_mask,
    })

    return token_features
