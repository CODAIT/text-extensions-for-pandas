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

# Internal imports
from pandas_text.char_span import CharSpanArray
from pandas_text.token_span import TokenSpanArray

# Set to True to use sparse storage for tokens 2-n of n-token dictionary
# entries. First token is always stored dense, of course.
# Currently set to False to avoid spurious Pandas API warnings about conversion
# from sparse to dense.
# TODO: Turn this back on when Pandas fixes the issue with the warning.
_SPARSE_DICT_ENTRIES = False


def make_tokens(target_text: str,
                tokenizer: spacy.tokenizer.Tokenizer) -> pd.Series:
    """
    :param target_text: Text to tokenize
    :param tokenizer: Preconfigured tokenizer object
    :return: The tokens (and underlying text) as a Pandas Series wrapped around
        a `CharSpanArray` value.
    """
    spacy_doc = tokenizer(target_text)
    tok_begins = np.array([t.idx for t in spacy_doc])
    tok_ends = np.array([t.idx + len(t) for t in spacy_doc])
    return pd.Series(CharSpanArray(target_text, tok_begins, tok_ends))


def load_dict(file_name: str, tokenizer: spacy.tokenizer.Tokenizer):
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
            toks_list if i == 0 or not _SPARSE_DICT_ENTRIES
            else pd.SparseArray(toks_list)
        )

    return pd.DataFrame(cols_dict)


def extract_dict(tokens: Union[CharSpanArray, pd.Series],
                 dictionary: pd.DataFrame,
                 target_col_name: str = "matches"):
    """
    Identify all matches of a dictionary on a sequence of tokens.

    :param tokens: `CharSpanArray` of token information, optionally wrapped in a
    `pd.Series`.

    :param dictionary: The dictionary to match, encoded as a `pd.DataFrame` in
    the format returned by `load_dict()`

    :param target_col_name: (optional) name of column of matching spans in the
    returned DataFrame

    :return: a single-column DataFrame of token ID spans of dictionary matches
    """
    # Box tokens into a pd.Series if not already boxed.
    if isinstance(tokens, CharSpanArray):
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
    return pd.DataFrame({target_col_name: TokenSpanArray(tokens.values,
                                                         begins, ends)})


def extract_regex_tok(
        tokens: Union[CharSpanArray, pd.Series],
        compiled_regex: regex.Regex,
        min_len=1,
        max_len=1):
    """
    Identify all (possibly overlapping) matches of a regular expression
    that start and end on token boundaries.

    :param tokens: `CharSpanArray` of token information, optionally wrapped in a
    `pd.Series`.
    :param compiled_regex: Regular expression to evaluate.
    :param min_len: Minimum match length in tokens
    :param max_len: Maximum match length (inclusive) in tokens

    :returns: A single-column DataFrame containing a span for each match of the
    regex.
    """
    if isinstance(tokens, CharSpanArray):
        tokens = pd.Series(tokens)

    num_tokens = len(tokens.values)
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

        window_tok_spans = TokenSpanArray(tokens.values, window_begin_toks,
                                          window_end_toks)
        matches_list.append(pd.Series(
            window_tok_spans[matches_regex_f(window_tok_spans.covered_text)]
        ))
    return pd.DataFrame({"matches": pd.concat(matches_list)})


def adjacent_join(first_series: pd.Series,
                  second_series: pd.Series,
                  first_name: str = "first",
                  second_name: str = "second",
                  min_gap: int = 0,
                  max_gap: int = 0):
    """
    Compute the join of two series of spans, where a pair of spans is
    considered to match if they are adjacent to each other in the text.

    :param first_series: Spans that appear earlier. dtype must be TokenSpan.

    :param second_series: Spans that come after. dtype must be TokenSpan.

    :param first_name: Name to give the column in the returned dataframe that
    is derived from `first_series`.

    :param second_name: Column name for spans from `second_series` in the
    returned DataFrame.

    :param min_gap: Minimum number of spans allowed between matching pairs of
    spans, inclusive.

    :param max_gap: Maximum number of spans allowed between matching pairs of
    spans, inclusive.

    :returns: a new `pd.DataFrame` containing all pairs of spans that match
    the join predicate. Columns of the DataFrame will be named according
    to the `first_name` and `second_name` arguments.
    """
    # For now we always make the first series the outer.
    # TODO: Make the larger series the outer and adjust the join logic
    # below accordingly.
    outer = pd.DataFrame({
            "outer_span": first_series,
            "outer_end": first_series.values.end_token
    })

    # Inner series gets replicated for every possible offset so we can use
    # Pandas' high-performance equijoin
    inner_span_list = [second_series] * (max_gap - min_gap + 1)
    outer_end_list = [
        # Join predicate: outer_span = inner_span.begin + gap
        second_series.values.begin_token + gap
        for gap in range(min_gap, max_gap + 1)
    ]
    inner = pd.DataFrame({
        "inner_span": pd.concat(inner_span_list),
        "outer_end": np.concatenate(outer_end_list)
    })
    joined = outer.merge(inner)

    # Now we have a DataFrame with the schema
    # [outer_span, outer_end, inner_span]
    return pd.DataFrame({
        first_name: joined["outer_span"],
        second_name: joined["inner_span"]
    })


def combine_spans(series1: pd.Series, series2: pd.Series):
    """
    :param series1: A series backed by a TokenSpanArray
    :param series2: A series backed by a TokenSpanArray
    :return: A new series (also backed by a TokenSpanArray) of spans
        containing shortest span that completely covers both input spans.
    """
    spans1 = series1.values
    spans2 = series2.values
    if not isinstance(spans1, TokenSpanArray) or not isinstance(spans2,
                                                               TokenSpanArray):
        raise ValueError("This function is only implemented for TokenSpanArrays")
    # TODO: Raise an error if any span in series1 comes after the corresponding
    #  span in series2
    # TODO: Raise an error if series1.tokens != series2.tokens
    return pd.Series(TokenSpanArray(
        spans1.tokens, spans1.begin_token, spans2.end_token
    ))
