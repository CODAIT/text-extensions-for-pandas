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
# extract.py
#
# Variants of the Extract operator from spanner algebra. The Extract operator
# returns sub-spans of a parent span that match a predicate.
#

import numpy as np
import pandas as pd
import regex

from typing import *

# Internal imports
from text_extensions_for_pandas.array.span import (
    SpanArray
)
from text_extensions_for_pandas.array.token_span import (
    TokenSpanArray
)

# Set to True to use sparse storage for tokens 2-n of n-token dictionary
# entries. First token is always stored dense, of course.
# Currently set to False to avoid spurious Pandas API warnings about conversion
# from sparse to dense.
# TODO: Turn this back on when Pandas fixes the issue with the warning.
_SPARSE_DICT_ENTRIES = False


def load_dict(file_name: str, tokenizer: "spacy.tokenizer.Tokenizer"):
    """
    Load a SystemT-format dictionary file. File format is one entry per line.

    Tokenizes and normalizes the dictionary entries.

    :param file_name: Path to dictionary file

    :param tokenizer: Preconfigured tokenizer object for tokenizing
    dictionary entries.  **Must be the same configuration as the tokenizer
    used on the target text!**

    :return: a `pd.DataFrame` with the normalized entries.
    """
    with open(file_name, "r") as f:
        lines = [
            line.strip() for line in f.readlines() if len(line) > 0 and line[0] != "#"
        ]

    # Tokenize with SpaCy. Produces a SpaCy document object per line.
    tokenized_entries = [tokenizer(line.lower()) for line in lines]

    # Determine the number of tokens in the longest dictionary entry.
    max_num_toks = max([len(e) for e in tokenized_entries])

    # Generate a column for each token. Go one past the max number of tokens so
    # that every dictionary entry ends up None-terminated.
    cols_dict = {}
    for i in range(max_num_toks + 1):
        # Extract token i from every entry that has a token i
        toks_list = [e[i].text if len(e) > i else None for e in tokenized_entries]
        cols_dict["toks_{}".format(i)] = (
            # Sparse storage for tokens 2 and onward
            toks_list
            if i == 0 or not _SPARSE_DICT_ENTRIES
            else pd.SparseArray(toks_list)
        )

    return pd.DataFrame(cols_dict)


def extract_dict(tokens: Union[SpanArray, pd.Series],
                 dictionary: pd.DataFrame,
                 output_col_name: str = "match"):
    """
    Identify all matches of a dictionary on a sequence of tokens.

    :param tokens: `SpanArray` of token information, optionally wrapped in a
    `pd.Series`.

    :param dictionary: The dictionary to match, encoded as a `pd.DataFrame` in
    the format returned by `load_dict()`

    :param output_col_name: (optional) name of column of matching spans in the
    returned DataFrame

    :return: a single-column DataFrame of token ID spans of dictionary matches
    """
    # Box tokens into a pd.Series if not already boxed.
    if isinstance(tokens, SpanArray):
        tokens = pd.Series(tokens)

    # Wrap the important parts of the tokens series in a temporary dataframe.
    toks_tmp = pd.DataFrame({
        "token_id": tokens.index,
        "normalized_text": tokens.values.normalized_covered_text
    })

    # Start by matching the first token.
    matches = pd.merge(dictionary, toks_tmp,
                       left_on="toks_0", right_on="normalized_text")
    matches.rename(columns={"token_id": "begin_token_id"}, inplace=True)
    matches_col_names = list(matches.columns)  # We'll need this later

    # Check against remaining elements of matching dictionary entries and
    # accumulate the full set of matches as a list of IntervalIndexes
    begins_list = []
    ends_list = []
    max_entry_len = len(dictionary.columns)
    for match_len in range(1, max_entry_len):
        # print("Match len: {}".format(match_len))
        # Find matches of length match_len. Dictionary entries of this length
        # will have None in the column "toks_<match_len>".
        match_locs = pd.isna(matches["toks_{}".format(match_len)])
        # print("Completed matches:\n{}".format(matches[match_locs]))
        match_begins = matches[match_locs]["begin_token_id"].to_numpy()
        match_ends = match_begins + match_len
        begins_list.append(match_begins)
        ends_list.append(match_ends)

        # For the remaining partial matches against longer dictionary entries,
        # check the next token by merging with the tokens dataframe.
        potential_matches = matches[~match_locs].copy()
        # print("Raw potential matches:\n{}".format(potential_matches))
        potential_matches.drop("normalized_text", axis=1, inplace=True)
        potential_matches["next_token_id"] = potential_matches[
                                                 "begin_token_id"] + match_len
        potential_matches = pd.merge(potential_matches, toks_tmp,
                                     left_on="next_token_id",
                                     right_on="token_id")
        # print("Filtered potential matches:\n{}".format(potential_matches))
        potential_matches = potential_matches[
            potential_matches["normalized_text"] == potential_matches[
                "toks_{}".format(match_len)]]
        # The result of the join has some extra columns that we don't need.
        matches = potential_matches[matches_col_names]
    # Gather together all the sets of matches and wrap in a dataframe.
    begins = np.concatenate(begins_list)
    ends = np.concatenate(ends_list)
    result = pd.DataFrame({output_col_name: TokenSpanArray(tokens.values,
                                                           begins, ends)})
    # Results are sorted by number of tokens; sort by location instead.
    result["__begin"] = result[output_col_name].values.begin
    return result.sort_values("__begin")[[output_col_name]]


def extract_regex_tok(
        tokens: Union[SpanArray, pd.Series],
        compiled_regex: regex.Regex,
        min_len=1,
        max_len=1,
        output_col_name: str = "match"):
    """
    Identify all (possibly overlapping) matches of a regular expression
    that start and end on token boundaries.

    :param tokens: `SpanArray` of token information, optionally wrapped in a
    `pd.Series`.

    :param compiled_regex: Regular expression to evaluate.

    :param min_len: Minimum match length in tokens

    :param max_len: Maximum match length (inclusive) in tokens

    :param output_col_name: (optional) name of column of matching spans in the
    returned DataFrame

    :returns: A single-column DataFrame containing a span for each match of the
    regex.
    """
    if isinstance(tokens, pd.Series):
        tokens = tokens.values

    num_tokens = len(tokens)
    matches_regex_f = np.vectorize(lambda s: compiled_regex.fullmatch(s)
                                   is not None)

    # The built-in regex functionality of Pandas/Python does not have
    # an optimized single-pass RegexTok, so generate all the places
    # where there might be a match and run them through regex.fullmatch().
    # Note that this approach is asymptotically inefficient if max_len is large.
    # TODO: Performance tuning for both small and large max_len
    matches_list = []
    for cur_len in range(min_len, max_len + 1):
        window_begin_toks = np.arange(0, num_tokens - cur_len + 1)
        window_end_toks = window_begin_toks + cur_len

        window_tok_spans = TokenSpanArray(tokens, window_begin_toks,
                                          window_end_toks)
        matches_list.append(pd.Series(
            window_tok_spans[matches_regex_f(window_tok_spans.covered_text)]
        ))
    return pd.DataFrame({output_col_name: pd.concat(matches_list)})
