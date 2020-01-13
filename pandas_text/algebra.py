################################################################################
# algebra.py
#
# Span manipulation functions from pandas_text


import pandas as pd
import numpy as np
import regex
import spacy
from typing import *
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from pandas_text import CharSpanArray, TokenSpanArray


def make_tokens(target_text: str,
                tokenizer: spacy.tokenizer.Tokenizer) -> CharSpanArray:
    """
    :param target_text: Text to tokenize
    :param tokenizer: Preconfigured tokenizer object
    :return: The tokens (and underlying text) as a `CharSpanArray` that can
        serve as a Pandas series or as the tokens object for a `TokenSpanArray`
    """
    spacy_doc = tokenizer(target_text)
    tok_begins = np.array([t.idx for t in spacy_doc])
    tok_ends = np.array([t.idx + len(t) for t in spacy_doc])
    return CharSpanArray(target_text, tok_begins, tok_ends)


def load_dict(file_name: str, tokenizer: spacy.tokenizer.Tokenizer):
    """
    Load a SystemT-format dictionary file. File format is one entry per line.

    Tokenizes and normalizes the dictionary entries.

    Returns a `pd.DataFrame` with the normalized entries.
    """
    with open(file_name, "r") as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != "#"]

    # Tokenize with SpaCy. Produces a SpaCy document object per line.
    tokenized_entries = [tokenizer(l.lower()) for l in lines]

    # Determine the number of tokens in the longest dictionary entry.
    max_num_toks = max([len(e) for e in tokenized_entries])

    # Generate a column for each token. Go one past the max number of tokens so
    # that every dictionary entry ends up None-terminated.
    cols_dict = {}
    for i in range(max_num_toks + 1):
        # Extract token i from every entry that has a token i
        toks_list = [e[i].text if len(e) > i else None for e in
                     tokenized_entries]
        cols_dict["toks_{}".format(i)] = (
            # Sparse storage for tokens 2 and onward
            toks_list if i == 0 else pd.SparseArray(toks_list)
        )

    return pd.DataFrame(cols_dict)


def extract_dict(tokens: pd.DataFrame, dictionary: pd.DataFrame,
                 target_col_name: str = "matches"):
    """
    Identify all matches of a dictionary on a sequence of tokens.

    Args:
        tokens: Dataframe of token information. Must have the the fields `token_id`,
                `text`, and `char_offsets`.
        dictionary: The dictionary to match, encoded as a `pd.DataFrame`.
        target_col_name: (optional) name of column of matching spans in returned
                Dataframe

    Returns a single-column dataframe of token ID spans of dictionary matches
    """
    # Generate and cache normalized tokens if not present
    if "normalized_text" not in tokens:
        tokens["normalized_text"] = tokens["text"].str.lower()

    # Match first token to find potential matches
    matches = pd.merge(dictionary,
                       tokens[["token_id", "normalized_text"]],
                       left_on="toks_0", right_on="normalized_text")
    matches.rename(columns={"token_id": "begin_token_id"}, inplace=True)
    matches_col_names = list(matches.columns)  # We'll need this later

    # Check against remaining elements of matching dictionary entries and
    # accumulate the full set of matches as a list of IntervalIndexes
    all_matches = []
    max_entry_len = len(dictionary.columns)
    for match_len in range(1, max_entry_len):
        # print("Match len: {}".format(match_len))
        # Find matches of length match_len. Dictionary entries of this length
        # will have None in the column "toks_<match_len>".
        match_locs = pd.isna(matches["toks_{}".format(match_len)])
        # print("Completed matches:\n{}".format(matches[match_locs]))
        match_begins = matches[match_locs]["begin_token_id"].to_numpy()
        match_ends = match_begins + match_len
        cur_spans = pd.IntervalIndex.from_arrays(match_begins, match_ends,
                                                 closed="left")
        all_matches.append(cur_spans)

        # For the remaining partial matches against longer dictionary entries,
        # check the next token by merging with the tokens dataframe.
        potential_matches = matches[~match_locs].copy()
        # print("Raw potential matches:\n{}".format(potential_matches))
        potential_matches.drop("normalized_text", axis=1, inplace=True)
        potential_matches["next_token_id"] = potential_matches[
                                                 "begin_token_id"] + match_len
        potential_matches = pd.merge(potential_matches,
                                     tokens[["token_id", "normalized_text"]],
                                     left_on="next_token_id",
                                     right_on="token_id")
        # print("Filtered potential matches:\n{}".format(potential_matches))
        potential_matches = potential_matches[
            potential_matches["normalized_text"] == potential_matches[
                "toks_{}".format(match_len)]]
        # The result of the join has some extra columns that we don't need.
        matches = potential_matches[matches_col_names]
    all_matches_as_df = [pd.DataFrame({target_col_name: m}) for m in
                         all_matches]
    return pd.concat(all_matches_as_df)


def extract_regex_tok(
        token_offsets: pd.Series,
        target_str: str,
        compiled_regex: regex.Regex,
        min_len=1,
        max_len=1):
    """
    Identify all (possibly overlapping) matches of a regular expression
    that start and end on token boundaries.

    Arguments:
        token_offsets: Series of spans that mark the locations of tokens
                       in the document
        target_str: Text of the document
        compiled_regex: Regular expression to evaluate.
        min_len: Minimum match length in tokens
        max_len: Maximum match length (inclusive) in tokens
    """
    token_begins = pd.arrays.IntervalArray(token_offsets).left
    token_ends = pd.arrays.IntervalArray(token_offsets).right
    num_tokens = token_begins.size

    # The built-in regex functionality of Pandas/Python does not have
    # an optimized single-pass RegexTok, so generate all the places
    # where there might be a match and run them through regex.fullmatch().
    # Note that this is inefficient if max_len is large.
    # TODO: Either directly implement token-based matching in C++ or
    # fall back on a scalable Python implementation
    matches_list = []
    for cur_len in range(min_len, max_len + 1):
        window_begin_toks = np.arange(0, num_tokens - cur_len + 1)
        window_end_toks = window_begin_toks + cur_len
        window_tok_intervals = pd.Series(pd.arrays.IntervalArray.from_arrays(
            window_begin_toks, window_end_toks, closed="left"))

        # Compute window boundaries in characters directly from the tokens
        window_begin_chars = token_begins[:num_tokens - cur_len + 1]
        window_end_chars = token_ends[cur_len - 1:]
        window_char_intervals = pd.Series(pd.arrays.IntervalArray.from_arrays(
            window_begin_chars, window_end_chars, closed="left"))
        matching_windows = window_char_intervals.apply(
            lambda x: compiled_regex.fullmatch(target_str, x.left,
                                               x.right) is not None)
        matches_list.append(window_tok_intervals[matching_windows])

    return pd.DataFrame(
        {"matches": pd.concat(matches_list)})


def adjacent_join(first_series: pd.Series,
                  second_series: pd.Series,
                  first_name: str = "first",
                  second_name: str = "second",
                  min_gap: int = 0,
                  max_gap: int = 0):
    """
    Compute the join of two series of spans, where a pair of spans is
    considered to match if they are adjacent to each other in the text.

    Args:
        first_series: Spans that appear earlier. Offsets in tokens, not
                      characters.
        second_series: Spans that come after. Offsets in tokens.
        first_name: Name to give the column in the returned dataframe that
                    is derived from `first_series`.
        second_name: Column name for spans from `second_series` in the
                     returned dataframe.
        min_gap: Minimum number of spans allowed between matching pairs of
                 spans, inclusive.
        max_gap: Maximum number of spans allowed between matching pairs of
                 spans, inclusive.

    Returns a new `pd.DataFrame` containing all pairs of spans that match
    the join predicate. Columns of the dataframe will be named according
    to the `first_name` and `second_name` arguments.
    """
    # For now we always make the first series the outer.
    # TODO: Make the larger series the outer and adjust the join logic
    # below accordingly.
    outer = pd.DataFrame()
    outer["outer_span"] = first_series
    outer["outer_end"] = pd.IntervalIndex(first_series).right

    # Inner series gets replicated for every possible offset so we can use
    # Pandas' high-performance equijoin
    inner_chunks = []
    for gap in range(min_gap, max_gap + 1):
        chunk = pd.DataFrame()
        chunk["inner_span"] = second_series
        # Join predicate: outer.end == inner.begin + gap
        chunk["outer_end"] = pd.arrays.IntervalArray(second_series).left + gap
        inner_chunks.append(chunk)
    inner = pd.concat(inner_chunks)
    joined = outer.merge(inner)

    # Now we have a dataframe with the schema
    # [outer_span, outer_end, inner_span]
    ret = pd.DataFrame()
    ret[first_name] = joined["outer_span"]
    ret[second_name] = joined["inner_span"]
    return ret
