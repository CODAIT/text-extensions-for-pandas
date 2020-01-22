################################################################################
# algebra.py
#
# Span manipulation functions from pandas_text


import pandas as pd
import numpy as np
import regex
import spacy
import spacy.tokens.doc
from typing import *
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

# Internal imports
from pandas_text.char_span import CharSpan, CharSpanType, CharSpanArray
from pandas_text.token_span import TokenSpan, TokenSpanType, TokenSpanArray

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


def make_tokens_and_features(
    target_text: str,
    language_model: spacy.language.Language) -> pd.DataFrame:
    """
    :param target_text: Text to analyze
    :param language_model: Preconfigured spaCy language model object
    :return: The tokens of the text plus additional linguistic features that the
    language model generates, represented as a `pd.DataFrame`.
    """
    spacy_doc = language_model(target_text)
    # TODO: Performance tuning of the translation code that follows
    # Represent the character spans of the tokens
    tok_begins = np.array([t.idx for t in spacy_doc])
    tok_ends = np.array([t.idx + len(t) for t in spacy_doc])
    tokens_array = CharSpanArray(target_text, tok_begins, tok_ends)
    tokens_series = pd.Series(tokens_array)

    # Also build token-based spans to make it easier to compose
    token_spans = TokenSpanArray.from_char_offsets(tokens_series.values)

    # spaCy identifies tokens by semi-arbitrary integer "indexes" (in practice,
    # the offset of the first character in the token). Translate from these
    # to a dense range of integer IDs that will correspond to the index of our
    # returned DataFrame.
    idx_to_id = {spacy_doc[i].idx: i for i in range(len(spacy_doc))}

    return pd.DataFrame({
        "token_num": range(len(tok_begins)),
        "char_span": tokens_series,
        "token_span": token_spans,
        "lemma": [t.lemma_ for t in spacy_doc],
        "pos": pd.Categorical([t.pos_ for t in spacy_doc]),
        "tag": pd.Categorical([t.tag_ for t in spacy_doc]),
        "dep": pd.Categorical([t.dep_ for t in spacy_doc]),
        "head_token_num": pd.Categorical([idx_to_id[t.head.idx]
                                          for t in spacy_doc]),
        "shape": pd.Categorical([t.shape_ for t in spacy_doc]),
        "is_alpha": np.array([t.is_alpha for t in spacy_doc]),
        "is_stop": np.array([t.is_stop for t in spacy_doc]),
        "sentence": _make_sentences_series(spacy_doc, tokens_array)
    })


def _make_sentences_series(spacy_doc: spacy.tokens.doc.Doc,
                           tokens: CharSpanArray):
    """
    Subroutine of `make_tokens_and_features()`

    :param spacy_doc: parsed document from a spaCy language model

    :param tokens: Token information for the current document as a
    `CharSpanArray` object. Must contain the same tokens as `spacy_doc`.

    :return: a Pandas DataFrame Series containing the token span of the (single)
    sentence that the token is in
    """
    num_toks = len(spacy_doc)
    # Generate the [begin, end) intervals that make up a series of spans
    begin_tokens = np.full(shape=num_toks, fill_value=-1, dtype=np.int)
    end_tokens = np.full(shape=num_toks, fill_value=-1, dtype=np.int)
    for sent in spacy_doc.sents:
        begin_tokens[sent.start:sent.end] = sent.start
        end_tokens[sent.start:sent.end] = sent.end
    return pd.Series(TokenSpanArray(tokens, begin_tokens, end_tokens))


def token_features_to_tree(token_features: pd.DataFrame,
                           text_col: str = "token_span",
                           tag_col: str = "tag",
                           label_col: str = "dep"):
    """
    Convert a DataFrame in the format returned by `make_tokens_and_features()`
    to the public input format of displaCy's dependency tree renderer.

    :param token_features: A subset of a token features DataFrame in the format
    returned by `make_tokens_and_features()`. Must at a minimum contain the
    `head_token_num` column and an integer index that corresponds to the ints
    in the `head_token_num` column.

    :param text_col: Name of the column in `token_features` from which the
    'covered text' label for each node of the parse tree should be extracted,
    or `None` to leave those labels blank.

    :param tag_col: Name of the column in `token_features` from which the
    'tag' label for each node of the parse tree should be extracted; or `None`
    to leave those labels blank.

    :param label_col: Name of the column in `token_features` from which the
    label for each edge of the parse tree should be extracted; or `None`
    to leave those labels blank.

    :returns: Native Python type representation of the parse tree in a format
    suitable to pass to `displacy.render(manual=True ...)`
    See https://spacy.io/usage/visualizers for the specification of this format.
    """

    # displaCy expects most inputs as strings. Centralize this conversion.
    def _get_text(col_name):
        if col_name is None:
            return np.zeros(shape=len(token_features.index), dtype=str)
        series = token_features[col_name]
        if isinstance(series.dtype, (CharSpanType, TokenSpanType)):
            return series.values.covered_text
        else:
            return series.astype(str)

    # Renumber the head_token_num column to a dense range starting from zero
    tok_map = {token_features.index[i]: i
               for i in range(len(token_features.index))}
    # Note that we turn any links to tokens not in our input rows into
    # self-links, which will get removed later on.
    head_tok = token_features["head_token_num"].values
    remapped_head_tok = []
    for i in range(len(token_features.index)):
        remapped_head_tok.append(
            tok_map[head_tok[i]] if head_tok[i] in tok_map
            else i
        )

    words_df = pd.DataFrame({
        "text": _get_text(text_col),
        "tag": _get_text(tag_col)
    })
    edges_df = pd.DataFrame({
        "from": range(len(token_features.index)),
        "to": remapped_head_tok,
        "label": _get_text(label_col),
    })
    # displaCy requires all arcs to have their start and end be in
    # numeric order. An additional attribute "dir" tells which way
    # (left or right) each arc goes.
    arcs_df = pd.DataFrame({
        "start": edges_df[["from", "to"]].min(axis=1),
        "end": edges_df[["from", "to"]].max(axis=1),
        "label": edges_df["label"],
        "dir": "left"
    })
    arcs_df["dir"].mask(edges_df["from"] > edges_df["to"], "right",
                        inplace=True)

    # Don't render self-links
    arcs_df = arcs_df[arcs_df["start"] != arcs_df["end"]]

    return {
        "words": words_df.to_dict(orient="records"),
        "arcs": arcs_df.to_dict(orient="records")
    }


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
        lines = [line.strip() for line in f.readlines() if len(line) > 0
                 and line[0] != "#"]

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
                 output_col_name: str = "match"):
    """
    Identify all matches of a dictionary on a sequence of tokens.

    :param tokens: `CharSpanArray` of token information, optionally wrapped in a
    `pd.Series`.

    :param dictionary: The dictionary to match, encoded as a `pd.DataFrame` in
    the format returned by `load_dict()`

    :param output_col_name: (optional) name of column of matching spans in the
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
    return pd.DataFrame({output_col_name: TokenSpanArray(tokens.values,
                                                         begins, ends)})


def extract_regex_tok(
    tokens: Union[CharSpanArray, pd.Series],
    compiled_regex: regex.Regex,
    min_len=1,
    max_len=1,
    output_col_name: str = "match"):
    """
    Identify all (possibly overlapping) matches of a regular expression
    that start and end on token boundaries.

    :param tokens: `CharSpanArray` of token information, optionally wrapped in a
    `pd.Series`.

    :param compiled_regex: Regular expression to evaluate.

    :param min_len: Minimum match length in tokens

    :param max_len: Maximum match length (inclusive) in tokens

    :param output_col_name: (optional) name of column of matching spans in the
    returned DataFrame

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
    return pd.DataFrame({output_col_name: pd.concat(matches_list)})


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
        raise ValueError(
            "This function is only implemented for TokenSpanArrays")
    # TODO: Raise an error if any span in series1 comes after the corresponding
    #  span in series2
    # TODO: Raise an error if series1.tokens != series2.tokens
    return pd.Series(TokenSpanArray(
        spans1.tokens, spans1.begin_token, spans2.end_token
    ))
