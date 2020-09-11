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
# spacy.py
#
"""
This module contains I/O functions related to the SpaCy NLP library.
"""

import numpy as np
import pandas as pd

# To avoid creating an unnecessary dependency on SpaCy for non-SpaCy
# applications, we do NOT `import spacy` at the top level of this file,
# and we do NOT include type hints for SpaCy types in the function
# signatures below.

from text_extensions_for_pandas.array.span import (
    SpanArray,
    SpanDtype,
)
from text_extensions_for_pandas.array.token_span import (
    TokenSpanArray,
    TokenSpanDtype,
)


def make_tokens(target_text: str, tokenizer) -> pd.Series:
    """
    :param target_text: Text to tokenize
    :param tokenizer: Preconfigured `spacy.tokenizer.Tokenizer` object
    :return: The tokens (and underlying text) as a Pandas Series wrapped around
        a `SpanArray` value.
    """
    spacy_doc = tokenizer(target_text)
    tok_begins = np.array([t.idx for t in spacy_doc])
    tok_ends = np.array([t.idx + len(t) for t in spacy_doc])
    return pd.Series(SpanArray(target_text, tok_begins, tok_ends))


def make_tokens_and_features(
    target_text: str, language_model, add_left_and_right=False,
) -> pd.DataFrame:
    """
    :param target_text: Text to analyze

    :param language_model: Preconfigured spaCy language model (`spacy.language.Language`)
     object

    :param add_left_and_right: If `True`, add columns "left" and "right"
    containing references to previous and next tokens.

    :return: A tuple of two dataframes:
    1. The tokens of the text plus additional linguistic features that the
       language model generates, represented as a `pd.DataFrame`.
    2. A table of named entities identified by the language model's named entity
       tagger, represented as a `pd.DataFrame`.
    """
    spacy_doc = language_model(target_text)

    # TODO: Performance tuning of the translation code that follows
    # Represent the character spans of the tokens
    tok_begins = np.array([t.idx for t in spacy_doc])
    tok_ends = np.array([t.idx + len(t) for t in spacy_doc])
    tokens_array = SpanArray(target_text, tok_begins, tok_ends)
    tokens_series = pd.Series(tokens_array)
    # Also build single-token token-based spans to make it easier to build
    # larger token-based spans.
    token_spans = TokenSpanArray.from_char_offsets(tokens_series.values)
    # spaCy identifies tokens by semi-arbitrary integer "indexes" (in practice,
    # the offset of the first character in the token). Translate from these
    # to a dense range of integer IDs that will correspond to the index of our
    # returned DataFrame.
    idx_to_id = {spacy_doc[i].idx: i for i in range(len(spacy_doc))}
    # Define the IOB categorical type with "O" == 0, "B"==1, "I"==2
    iob2_dtype = pd.CategoricalDtype(["O", "B", "I"], ordered=False)
    df_cols = {
        "id": range(len(tok_begins)),
        "span": tokens_series,
        "lemma": [t.lemma_ for t in spacy_doc],
        "pos": pd.Categorical([str(t.pos_) for t in spacy_doc]),
        "tag": pd.Categorical([str(t.tag_) for t in spacy_doc]),
        "dep": pd.Categorical([str(t.dep_) for t in spacy_doc]),
        "head": np.array([idx_to_id[t.head.idx] for t in spacy_doc]),
        "shape": pd.Categorical([t.shape_ for t in spacy_doc]),
        "ent_iob": pd.Categorical([str(t.ent_iob_) for t in spacy_doc],
                                  dtype=iob2_dtype),
        "ent_type": pd.Categorical([str(t.ent_type_) for t in spacy_doc]),
        "is_alpha": np.array([t.is_alpha for t in spacy_doc]),
        "is_stop": np.array([t.is_stop for t in spacy_doc]),
        "sentence": _make_sentences_series(spacy_doc, tokens_array),
    }
    if add_left_and_right:
        # Use nullable int type because these columns contain nulls
        df_cols["left"] = pd.array(
            [None] + list(range(len(tok_begins) - 1)), dtype=pd.Int32Dtype()
        )
        df_cols["right"] = pd.array(
            list(range(1, len(tok_begins))) + [None], dtype=pd.Int32Dtype()
        )
    return pd.DataFrame(df_cols)


def _make_sentences_series(spacy_doc, tokens: SpanArray):
    """
    Subroutine of `make_tokens_and_features()`

    :param spacy_doc: parsed document (`spacy.tokens.doc.Doc`) from a spaCy language
     model

    :param tokens: Token information for the current document as a
    `SpanArray` object. Must contain the same tokens as `spacy_doc`.

    :return: a Pandas DataFrame Series containing the token span of the (single)
    sentence that the token is in
    """
    num_toks = len(spacy_doc)
    # Generate the [begin, end) intervals that make up a series of spans
    begin_tokens = np.full(shape=num_toks, fill_value=-1, dtype=np.int)
    end_tokens = np.full(shape=num_toks, fill_value=-1, dtype=np.int)
    for sent in spacy_doc.sents:
        begin_tokens[sent.start : sent.end] = sent.start
        end_tokens[sent.start : sent.end] = sent.end
    return pd.Series(TokenSpanArray(tokens, begin_tokens, end_tokens))


def token_features_to_tree(
    token_features: pd.DataFrame,
    text_col: str = "span",
    tag_col: str = "tag",
    label_col: str = "dep",
    head_col: str = "head",
):
    """
    Convert a DataFrame in the format returned by `make_tokens_and_features()`
    to the public input format of displaCy's dependency tree renderer.

    :param token_features: A subset of a token features DataFrame in the format
    returned by `make_tokens_and_features()`. Must at a minimum contain the
    `head` column and an integer index that corresponds to the ints
    in the `head` column.

    :param text_col: Name of the column in `token_features` from which the
    'covered text' label for each node of the parse tree should be extracted,
    or `None` to leave those labels blank.

    :param tag_col: Name of the column in `token_features` from which the
    'tag' label for each node of the parse tree should be extracted; or `None`
    to leave those labels blank.

    :param label_col: Name of the column in `token_features` from which the
    label for each edge of the parse tree should be extracted; or `None`
    to leave those labels blank.

    :param head_col: Name of the column in `token_features` from which the
     head node of each parse tree node should be extracted.

    :returns: Native Python type representation of the parse tree in a format
    suitable to pass to `displacy.render(manual=True ...)`
    See https://spacy.io/usage/visualizers for the specification of this format.
    """

    # displaCy expects most inputs as strings. Centralize this conversion.
    def _get_text(col_name):
        if col_name is None:
            return np.zeros(shape=len(token_features.index), dtype=str)
        series = token_features[col_name]
        if isinstance(series.dtype, (SpanDtype, TokenSpanDtype)):
            return series.values.covered_text
        else:
            return series.astype(str)

    # Renumber the head column to a dense range starting from zero
    tok_map = {token_features.index[i]: i for i in range(len(token_features.index))}
    # Note that we turn any links to tokens not in our input rows into
    # self-links, which will get removed later on.
    head_tok = token_features[head_col].values
    remapped_head_tok = []
    for i in range(len(token_features.index)):
        remapped_head_tok.append(tok_map[head_tok[i]] if head_tok[i] in tok_map else i)

    words_df = pd.DataFrame({"text": _get_text(text_col), "tag": _get_text(tag_col)})
    edges_df = pd.DataFrame(
        {
            "from": range(len(token_features.index)),
            "to": remapped_head_tok,
            "label": _get_text(label_col),
        }
    )
    # displaCy requires all arcs to have their start and end be in
    # numeric order. An additional attribute "dir" tells which way
    # (left or right) each arc goes.
    arcs_df = pd.DataFrame(
        {
            "start": edges_df[["from", "to"]].min(axis=1),
            "end": edges_df[["from", "to"]].max(axis=1),
            "label": edges_df["label"],
            "dir": "left",
        }
    )
    arcs_df["dir"].mask(edges_df["from"] > edges_df["to"], "right", inplace=True)

    # Don't render self-links
    arcs_df = arcs_df[arcs_df["start"] != arcs_df["end"]]

    return {
        "words": words_df.to_dict(orient="records"),
        "arcs": arcs_df.to_dict(orient="records"),
    }


def render_parse_tree(
    token_features: pd.DataFrame,
    text_col: str = "span",
    tag_col: str = "tag",
    label_col: str = "dep",
    head_col: str = "head",
) -> None:
    """
    Display a DataFrame in the format returned by `make_tokens_and_features()`
    using displaCy's dependency tree renderer.
    See https://spacy.io/usage/visualizers for more information on displaCy.

    :param token_features: A subset of a token features DataFrame in the format
    returned by `make_tokens_and_features()`. Must at a minimum contain the
    `head` column and an integer index that corresponds to the ints
    in the `head` column.
    :param text_col: Name of the column in `token_features` from which the
    'covered text' label for each node of the parse tree should be extracted,
    or `None` to leave those labels blank.
    :param tag_col: Name of the column in `token_features` from which the
    'tag' label for each node of the parse tree should be extracted; or `None`
    to leave those labels blank.
    :param label_col: Name of the column in `token_features` from which the
    label for each edge of the parse tree should be extracted; or `None`
    to leave those labels blank.
    :param head_col: Name of the column in `token_features` from which the
     head node of each parse tree node should be extracted.
    """
    import spacy.displacy

    return spacy.displacy.render(
        token_features_to_tree(token_features, text_col, tag_col, label_col,
                               head_col),
        manual=True,
    )
