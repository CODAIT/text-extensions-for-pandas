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
from typing import *

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
    from transformers import PreTrainedTokenizerFast

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            "Tokenizer must be an instance of "
            "transformers.PreTrainedTokenizerFast that supports "
            "encode_plus with return_offsets_mapping=True."
        )
    tokenized_result = tokenizer.encode_plus(
        target_text, return_special_tokens_mask=True, return_offsets_mapping=True
    )

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
    special_tokens_mask = pd.Series(tokenized_result["special_tokens_mask"]).astype(
        "bool"
    )

    # Fill remaining special tokens to zero-length spans
    ends = offset_df["end"].fillna(method="ffill").astype("int32")
    begins = offset_df["begin"].mask(special_tokens_mask, other=ends).astype("int32")

    # Create char and token span arrays
    char_spans = CharSpanArray(target_text, begins, ends)
    token_spans = TokenSpanArray(
        char_spans, np.arange(len(char_spans)), np.arange(1, len(char_spans) + 1)
    )

    token_features = pd.DataFrame(
        {
            "id": special_tokens_mask.index,
            # Use values instead of series because different indexes
            "char_span": pd.Series(char_spans).values,
            "token_span": pd.Series(token_spans).values,
            "input_id": tokenized_result["input_ids"],
            "token_type_id": tokenized_result["token_type_ids"],
            "attention_mask": tokenized_result["attention_mask"],
            "special_tokens_mask": special_tokens_mask,
        }
    )

    return token_features


def seq_to_windows(
    seq: np.ndarray, overlap: int, non_overlap: int
) -> Dict[str, np.ndarray]:
    """
    Convert a variable-length sequence into a set of fixed length windows,
    adding padding as necessary.

    Usually this function is used to prepare batches of BERT tokens to feed
    to a BERT model.

    :param seq: Original variable length sequence, as a 1D numpy array
    :param overlap: How much overlap there should be between adjacent windows
    :param non_overlap: How much non-overlapping content between the overlapping
     regions there should be at the middle of each window?

    :returns: A dictionary with two entries:
      * `input_ids`: 2D `np.ndarray` of fixed-length windows
      * `attention_masks`: 2D `np.ndarray` of attention masks (1 for
        tokens that are NOT masked, 0 for tokens that are masked)
        to feed into your favorite BERT-like embedding generator.
    """
    if len(seq.shape) != 1:
        raise ValueError(f"Input array must be 1D; got shape {seq.shape}")
    window_length, pre_padding, post_padding = _compute_padding(
        len(seq), overlap, non_overlap
    )

    # First generate the windows as a padded flat arrays.
    padded_length = len(seq) + pre_padding + post_padding
    buf = np.zeros(shape=[padded_length], dtype=seq.dtype)
    buf[pre_padding : pre_padding + len(seq)] = seq

    mask_buf = np.zeros_like(buf, dtype=int)  # 0 == MASKED
    mask_buf[pre_padding : pre_padding + len(seq)] = 1  # 1 == NOT MASKED

    # Reshape the flat arrays into overlapping windows.
    num_windows = padded_length // (overlap + non_overlap)
    windows = np.zeros(shape=[num_windows, window_length], dtype=seq.dtype)
    masks = np.zeros(shape=[num_windows, window_length], dtype=int)
    for i in range(num_windows):
        start = i * (overlap + non_overlap)
        windows[i] = buf[start : start + window_length]
        masks[i] = mask_buf[start : start + window_length]
    return {"input_ids": windows, "attention_masks": masks}


def windows_to_seq(
    seq: np.ndarray, windows: np.ndarray, overlap: int, non_overlap: int
) -> Dict[str, np.ndarray]:
    """
    Inverse of `seq_to_windows()`.
    Convert fixed length windows with padding to a variable-length sequence
    that matches up with the original sequence from which the windows were
    computed.

    Usually this function is used to convert the outputs of a BERT model
    back to a format that aligns with the original tokens.

    :param seq: Original variable length sequence to align with,
      as a 1D numpy array
    :param windows: Windowed data to align with the original sequence.
      Usually this data is the result of applying a transformation to the
      output of `seq_to_windows()`
    :param overlap: How much overlap there is between adjacent windows
    :param non_overlap: How much non-overlapping content between the overlapping
     regions there should be at the middle of each window?

    :returns: A 1D `np.ndarray` containing the contents of `windows` that
     correspond to the elements of `seq`.
    """
    if len(seq.shape) != 1:
        raise ValueError(f"Input array must be 1D; got shape {seq.shape}")
    window_length, pre_padding, post_padding = _compute_padding(
        len(seq), overlap, non_overlap
    )

    # Input may be an array of n-dimensional tensors.
    result_shape = [len(seq)] + list(windows.shape[2:])
    # result = np.zeros_like(seq, dtype=windows.dtype)
    result = np.zeros(shape=result_shape, dtype=windows.dtype)

    # Special-case the first and last windows.
    if len(seq) <= non_overlap + (overlap // 2):
        # Only one window, potentially a partial window.
        return windows[0][overlap : overlap + len(seq)]
    else:
        result[0 : non_overlap + (overlap // 2)] = windows[0][
            overlap : overlap + non_overlap + (overlap // 2)
        ]

    num_to_copy_from_last = overlap // 2 + overlap + non_overlap - post_padding
    if num_to_copy_from_last > 0:
        result[-num_to_copy_from_last:] = windows[-1][
            overlap // 2 : (overlap // 2) + num_to_copy_from_last
        ]

    # Remaining windows can be covered in a loop
    for i in range(1, len(windows) - 1):
        src_start = overlap // 2
        dest_start = overlap // 2 + non_overlap + (i - 1) * (overlap + non_overlap)
        num_to_copy = min(non_overlap + overlap, len(seq) - dest_start)
        result[dest_start : dest_start + num_to_copy] = windows[i][
            src_start : src_start + num_to_copy
        ]

    return result


def _compute_padding(
    seq_len: int, overlap: int, non_overlap: int
) -> Tuple[int, int, int]:
    """
    Shared padding computation for seq_to_windows() and windows_to_seq()

    :param seq_len: Length of original sequence
    :param overlap: How much overlap there should be between adjacent window
    :param non_overlap: How much non-overlapping content between the overlapping
     regions there should be at the middle of each window?

    :returns: A tuple of (window_length, pre_padding, post_padding)
    """
    if 0 != overlap % 2:
        raise ValueError(f"Non-even overlaps not implemented; got {overlap}")

    # Each window has overlapping regions at the beginning and end
    window_length = (2 * overlap) + non_overlap

    # Account for the padding before the first window
    pre_padding = overlap

    # Account for the padding after the last window
    remainder = (seq_len + pre_padding) % (overlap + non_overlap)
    post_padding = window_length - remainder
    if post_padding == window_length:
        # Chop off empty last window
        post_padding -= overlap + non_overlap

    return (window_length, pre_padding, post_padding)
