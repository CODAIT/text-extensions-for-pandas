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

"""
This module contains functions for working with transformer-based embeddings
such as BERT, including managing the special tokenization and windowing that these
embeddings require.

This module uses the ``transformers``_ library to implement tokenization and
embeddings. You will need that library in your Python path to use the
functions in this module.

.. _``transformers``: https://github.com/huggingface/transformers
"""

################################################################################
# tokenization.py
#
# Functions for tokenization of text

import numpy as np
import pandas as pd
from typing import *

from text_extensions_for_pandas.array.span import (
    SpanArray,
)
from text_extensions_for_pandas.array.token_span import (
    TokenSpanArray,
)
from text_extensions_for_pandas.array.tensor import (
    TensorArray,
)

from text_extensions_for_pandas.io import conll as conll
from text_extensions_for_pandas import spanner as spanner


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
     * "span": span of the token (with offsets measured in characters)
     * "input_id": integer ID suitable for input to a BERT embedding model
     * "token_type_id": list of token type ids to be fed to a model
     * "attention_mask": list of indices specifying which tokens should be
                         attended to by the model
     * "special_tokens_mask": `True` if the token is a zero-length special token
       such as "start of document"
    """
    # noinspection PyPackageRequirements
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

    spans = SpanArray(target_text, begins, ends)

    token_features = pd.DataFrame(
        {
            "token_id": special_tokens_mask.index,
            "span": spans,
            "input_id": tokenized_result["input_ids"],
            "token_type_id": tokenized_result["token_type_ids"],
            "attention_mask": tokenized_result["attention_mask"],
            "special_tokens_mask": special_tokens_mask,
        }
    )

    return token_features


def add_embeddings(df: pd.DataFrame, bert: Any,
                   overlap: int = 32, non_overlap: int = 64) -> pd.DataFrame:
    """
    Add BERT embeddings to a DataFrame of BERT tokens.

    :param df: DataFrame containing BERT tokens, as returned by
      :func:`make_bert_tokens` Must contain a column
      `input_id` containing token IDs.
    :param bert: PyTorch-based BERT model from the `transformers` library
    :param overlap: (optional) how much overlap there should be between adjacent windows
    :param non_overlap: (optional) how much non-overlapping content between the
     overlapping regions there should be at the middle of each window?
    :returns: A copy of `df` with a new column, "embedding" containing
     BERT embeddings as a `TensorArray`.

    .. note:: PyTorch must be installed to run this function.
    """
    # Import torch inline so that the rest of this library will function without it.
    # noinspection PyPackageRequirements
    import torch
    flat_input_ids = df["input_id"].values
    windows = seq_to_windows(flat_input_ids, overlap, non_overlap)
    bert_result = bert(
        input_ids=torch.tensor(windows["input_ids"]),
        attention_mask=torch.tensor(windows["attention_masks"]))
    hidden_states = windows_to_seq(flat_input_ids,
                                   bert_result[0].detach().numpy(),
                                   overlap, non_overlap)
    embeddings = TensorArray(hidden_states)
    ret = df.copy()
    ret["embedding"] = embeddings
    return ret


def conll_to_bert(df: pd.DataFrame, tokenizer: Any, bert: Any,
                  token_class_dtype: pd.CategoricalDtype,
                  compute_embeddings: bool = True,
                  overlap: int = 32, non_overlap: int = 64) -> pd.DataFrame:
    """
    :param df: One DataFrame from the conll_2003_to_dataframes() function,
     representing the tokens of a single document in the original tokenization.
    :param tokenizer: BERT tokenizer instance from the `transformers` library
    :param bert: PyTorch-based BERT model from the `transformers` library
    :param token_class_dtype: Pandas categorical type for representing
     token class labels, as returned by :func:`make_iob_tag_categories`
    :param compute_embeddings: True to generate BERT embeddings at each token
     position and add a column "embedding" to the returned DataFrame with
     the embeddings
    :param overlap: (optional) how much overlap there should be between adjacent
     windows for embeddings
    :param non_overlap: (optional) how much non-overlapping content between the
     overlapping regions there should be at the middle of each window?

    :returns: A version of the same DataFrame, but with BERT tokens, BERT
     embeddings for each token (if `compute_embeddings` is `True`),
     and token class labels.
    """
    spans_df = conll.iob_to_spans(df)
    bert_toks_df = make_bert_tokens(df["span"].values[0].target_text,
                                    tokenizer)
    bert_token_spans = TokenSpanArray.align_to_tokens(bert_toks_df["span"],
                                                      spans_df["span"])
    bert_toks_df[["ent_iob", "ent_type"]] = conll.spans_to_iob(bert_token_spans,
                                                               spans_df["ent_type"])
    bert_toks_df = conll.add_token_classes(bert_toks_df, token_class_dtype)
    if compute_embeddings:
        bert_toks_df = add_embeddings(bert_toks_df, bert, overlap, non_overlap)
    return bert_toks_df


def align_bert_tokens_to_corpus_tokens(
        spans_df: pd.DataFrame, corpus_toks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand entity matches from a BERT-based model so that they align
    with the corpus's original tokenization.

    :param spans_df: DataFrame of extracted entities. Must contain two
     columns: "span" and "ent_type". Other columns ignored.
    :param corpus_toks_df: DataFrame of the corpus's original tokenization,
     one row per token.
     Must contain a column "span" with character-based spans of
     the tokens.

    :returns: A new DataFrame with schema ["span", "ent_type"],
     where the "span" column contains token-based spans based off
     the *corpus* tokenization in `corpus_toks_df["span"]`.
    """
    if len(spans_df.index) == 0:
        return spans_df.copy()
    overlaps_df = (
        spanner
            .overlap_join(spans_df["span"], corpus_toks_df["span"],
                          "span", "corpus_token")
            .merge(spans_df)
    )
    agg_df = (
        overlaps_df
            .groupby("span")
            .aggregate({"corpus_token": "sum", "ent_type": "first"})
            .reset_index()
    )
    cons_df = (
        spanner.consolidate(agg_df, "corpus_token")
        [["corpus_token", "ent_type"]]
            .rename(columns={"corpus_token": "span"})
    )
    cons_df["span"] = TokenSpanArray.align_to_tokens(
        corpus_toks_df["span"], cons_df["span"])
    return cons_df


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
) -> np.ndarray:
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

    return window_length, pre_padding, post_padding
