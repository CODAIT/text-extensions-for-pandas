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
# conll.py
#
# I/O functions related to CONLL entity format and its many derivatives.

import pandas as pd
import numpy as np

from typing import *

from text_extensions_for_pandas.array import (
    CharSpanArray,
    TokenSpanArray,
    CharSpanType,
    TokenSpanType,
)


# Special token that CoNLL-2003 format uses to delineate the documents in
# the collection.
_CONLL_DOC_SEPARATOR = "-DOCSTART-"


def _parse_conll_file(input_file: str) -> List[List[Dict[str, List[str]]]]:
    """
    Parse the CoNLL-2003 file format for training/test data to Python
    objects.

    The format is especially tricky, so everything here is straight
    non-vectorized Python code. If you want performance, write the
    contents of your CoNLL files back out into a file format that
    supports performance.

    :param input_file: Location of the file to read
    :returns: A list of lists of dicts. The top list has one entry per
     document. The next level lists have one entry per sentence.
     Each sentence's dict contains lists under the following keys:
     * `token`: List of surface forms of tokens
     * `iob`: List of IOB tags as strings. This function does **NOT**
       correct for the silly way that CoNLL-format uses "B" tags. See
       `_fix_iob_tags()` for that correction.
     * `entity`: List of entity tags where `iob` contains I's or B's.
       `None` everywhere else.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Build up a list of document metadata as Python objects
    docs = []  # Type: List[List[Dict[str, List[str]]]]

    # Information about the current sentence
    tokens = []  # Type: List[str]
    iobs = []  # Type: List[str]
    entities = []  # Type: List[str]

    # Information about the current document
    sentences = []

    for i in range(len(lines)):
        l = lines[i].strip()
        if 0 == len(l):
            # Blank line is the sentence separator
            if len(tokens) > 0:
                sentences.append({"token": tokens,
                                  "iob": iobs,
                                  "entity": entities})
                tokens = []
                iobs = []
                entities = []
        else:
            # Not at the end of a sentence
            line_elems = l.split(" ")
            if len(line_elems) != 2:
                raise ValueError(f"Line {i} is not space-delimited.\n"
                                 f"Line was: '{l}'")
            token, raw_tag = line_elems
            if token == _CONLL_DOC_SEPARATOR:
                # End of previous document ==> start of next one
                docs.append(sentences)
                sentences = []
            else:
                # Not at the end of a sentence or a document. Must be
                # in the middle of a sentence.
                if raw_tag.startswith("I") or raw_tag.startswith("B"):
                    # Tokens that are entities are tagged with tags like
                    # "I-PER" or "B-MISC".
                    tag, entity = raw_tag.split("-")
                elif raw_tag == "O":
                    tag = raw_tag
                    entity = None
                else:
                    raise ValueError(f"Unexpected tag {raw_tag} at line {i}.\n"
                                     f"Line was: '{l}'")

                tokens.append(token)
                iobs.append(tag)
                entities.append(entity)

    # Close out the last token, sentence, and document, if needed
    if len(tokens) > 0:
        sentences.append({"token": tokens,
                          "iob": iobs,
                          "entity": entities})
    if len(sentences) > 0:
        docs.append(sentences)
    return docs


def _parse_conll_output_file(doc_dfs: List[pd.DataFrame],
                             input_file: str) -> List[Dict[str, List[str]]]:
    """
    Parse the CoNLL-2003 file format for output data to Python
    objects. This format is similar to the format that `_parse_conll_file`
    produces, but without the token and document boundary information.

    :param doc_dfs: List of `pd.DataFrame`s of token information from the
     corresponding training data file, one `DataFrame` per document.
     Used for determining document boundaries, which are not encoded in
     CoNLL-2003 output file format.
    :param input_file: Location of the file to read
    :returns: A list of dicts. The top list has one entry per
     document. The next level contains lists under the following keys:
     * `iob`: List of IOB tags as strings. This function does **NOT**
       correct for the silly way that CoNLL-format uses "B" tags. See
       `_fix_iob_tags()` for that correction.
     * `entity`: List of entity tags where `iob` contains I's or B's.
       `None` everywhere else.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Build up a list of document metadata as Python objects
    docs = []  # Type: List[Dict[str, List[str]]]

    # Position in the corpus
    doc_num = 0
    num_tokens_in_doc = len(doc_dfs[doc_num].index)
    token_num = 0

    # Information about the current document's tokens
    iobs = []  # Type: List[str]
    entities = []  # Type: List[str]

    for i in range(len(lines)):
        l = lines[i].strip()
        if 0 == len(l):
            # Blank line is the sentence separator.
            continue
        if " " in l:
            raise ValueError(f"Line {i} contains unexpected space character.\n"
                             f"Line was: '{l}'")
        raw_tag = l
        if raw_tag.startswith("I") or raw_tag.startswith("B"):
            # Tokens that are entities are tagged with tags like
            # "I-PER" or "B-MISC".
            tag, entity = raw_tag.split("-")
        elif raw_tag == "O":
            tag = raw_tag
            entity = None
        else:
            raise ValueError(f"Unexpected tag {raw_tag} at line {i}.\n"
                             f"Line was: '{l}'")
        iobs.append(tag)
        entities.append(entity)
        token_num += 1
        if token_num == num_tokens_in_doc:
            # End of current document, advance to next
            docs.append({
                "iob": iobs,
                "entity": entities
            })
            iobs = []
            entities = []
            doc_num += 1
            token_num = 0
            if doc_num < len(doc_dfs):
                num_tokens_in_doc = len(doc_dfs[doc_num].index)

    if doc_num < len(doc_dfs):
        raise ValueError(f"Corpus has {len(doc_dfs)} documents, but"
                         f"only found outputs for {doc_num} of them.")

    return docs


def _fix_iob_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    In CoNLL-2003 format, the first token of an entity is only tagged
    "B" when there are two entities of the same type back-to-back.
    Correct for that silliness by always tagging the first token
    with a "B"

    :param df: A `pd.DataFrame` that must contain the following two
     columns:
     * `ent_iob`: IOB tags (to be corrected)
     * `ent_type`: Entity type strings or `None` if a token is not
       an entity
     * `sentence`: Sentence spans

    :returns: A version of `df` with corrected IOB tags in the `ent_iob`
     column. The original dataframe is not modified.
    """
    iobs = df["ent_iob"].values.copy()  # Modified in place
    entities = df["ent_type"].values
    sentence_begins = df["sentence"].values.begin_token

    # Special-case the first one
    if iobs[0] == "I":
        iobs[0] = "B"
    for i in range(1, len(iobs)):
        tag = iobs[i]
        prev_tag = iobs[i - 1]
        if tag == "I":
            if (
                prev_tag == "O"  # Previous token not an entity
                or (prev_tag in ("I", "B")
                    and entities[i] != entities[i - 1]
            )  # Previous token a different type of entity
                or (sentence_begins[i] != sentence_begins[i-1]
            )  # Start of new sentence
            ):
                iobs[i] = "B"
    ret = df.copy()
    ret["ent_iob"] = iobs
    return ret

# def _fix_iob_tags__(iobs: np.ndarray, entities: np.ndarray):
#     """
#     In CoNLL-2003 format, the first token of an entity is only tagged
#     "B" when there are two entities of the same type back-to-back.
#     Correct for that silliness by always tagging the first token
#     with a "B"
#
#     Applies corrections in place to `iobs`
#
#     :param iobs: Array of IOB tags as strings. **Modified in place.**
#     :param entities; Array of entity tags, with `None` for the token
#      offsets with no entities.
#     """
#     # Special-case the first one
#     if iobs[0] == "I":
#         iobs[0] = "B"
#     for i in range(1, len(iobs)):
#         tag = iobs[i]
#         prev_tag = iobs[i - 1]
#         if tag == "I":
#             if (
#                 prev_tag == "O"  # Previous token not an entity
#                 or (prev_tag in ("I", "B")
#                     and entities[i] != entities[i - 1]
#             )  # Previous token a different type of entity
#             ):
#                 iobs[i] = "B"


def _doc_to_df(doc: List[Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Convert the "Python objects" representation of a document from a
    CoNLL-2003 file into a `pd.DataFrame` of token metadata.

    :param doc: Tree of Python objects that represents the document,
     List with one dictionary per sentence.
    :return: DataFrame with four columns:
    * `char_span`: Span of each token, with character offsets.
      Backed by the concatenation of the tokens in the document into
      a single string with one sentence per line.
    * `token_span`: Span of each token, with token offsets.
      Backed by the contents of the `char_span` column.
    * `ent_iob`: IOB-format tags of tokens, exactly as they appeared
      in the original file, with no corrections applied.
    * `ent_type`: Entity type names for tokens tagged "I" or "B" in
      the `ent_iob` column; `None` everywhere else.
    """
    begins_list = []  # Type: List[np.ndarray]
    ends_list = []  # Type: List[np.ndarray]
    sentences_list = []  # Type: List[np.ndarray]
    iobs_list = []  # Type: List[np.ndarray]
    entities_list = []  # Type: List[np.ndarray]
    sentence_begins_list = []  # Type: List[np.ndarray]
    sentence_ends_list = []  # Type: List[np.ndarray]

    char_position = 0
    token_position = 0
    for sentence in doc:
        sentence_text = " ".join(sentence["token"])
        sentences_list.append(sentence_text)
        lengths = np.array([len(t) for t in sentence["token"]])

        # Calculate begin and end offsets of tokens assuming 1 space
        # between them.
        e = np.cumsum(lengths + 1) - 1
        b = np.concatenate([[0], (e)[:-1] + 1])
        iobs = np.array(sentence["iob"])
        entities = np.array(sentence["entity"])
        sentence_begin_token = token_position
        sentence_end_token = token_position + len(e)
        sentence_begins = np.repeat(sentence_begin_token, len(e))
        sentence_ends = np.repeat(sentence_end_token, len(e))

        begins_list.append(b + char_position)
        ends_list.append(e + char_position)
        iobs_list.append(iobs)
        entities_list.append(entities)
        sentence_begins_list.append(sentence_begins)
        sentence_ends_list.append(sentence_ends)

        char_position += e[-1] + 1  # "+ 1" to account for newline
        token_position += len(e)

    begins = np.concatenate(begins_list)
    ends = np.concatenate(ends_list)
    doc_text = "\n".join(sentences_list)
    char_spans = CharSpanArray(doc_text, begins, ends)
    token_begins = np.arange(len(begins))
    token_spans = TokenSpanArray(char_spans, token_begins, token_begins + 1)
    sentence_spans = TokenSpanArray(char_spans,
                                    np.concatenate(sentence_begins_list),
                                    np.concatenate(sentence_ends_list))
    return pd.DataFrame(
        {"char_span": char_spans,
         "token_span": token_spans,
         "ent_iob": np.concatenate(iobs_list),
         "ent_type": np.concatenate(entities_list),
         "sentence": sentence_spans})


def _output_doc_to_df(tokens: pd.DataFrame,
                      outputs: Dict[str, List[str]],
                      copy_tokens: bool) -> pd.DataFrame:
    """
    Convert the "Python objects" representation of a document from a
    CoNLL-2003 file into a `pd.DataFrame` of token metadata.

    :param tokens: `pd.DataFrame` containing metadata about the tokens
     of this document, as returned by `conll_2003_to_dataframe`
    :param outputs: Dictionary containing outputs for this document,
     with fields "iob" and "entity".
    :param copy_tokens: `True` if token information should be deep-copied.
    :return: DataFrame with four columns:
    * `char_span`: Span of each token, with character offsets.
      Backed by the concatenation of the tokens in the document into
      a single string with one sentence per line.
    * `token_span`: Span of each token, with token offsets.
      Backed by the contents of the `char_span` column.
    * `ent_iob`: IOB-format tags of tokens, corrected so that every
      entity begins with a "B" tag.
    * `ent_type`: Entity type names for tokens tagged "I" or "B" in
      the `ent_iob` column; `None` everywhere else.
    """
    if copy_tokens:
        return pd.DataFrame(
            {"char_span": tokens["char_span"].copy(),
             "token_span": tokens["token_span"].copy(),
             "ent_iob": np.array(outputs["iob"]),
             "ent_type": np.array(outputs["entity"]),
             "sentence": tokens["sentence"].copy()})
    else:
        return pd.DataFrame(
            {"char_span": tokens["char_span"],
             "token_span": tokens["token_span"],
             "ent_iob": np.array(outputs["iob"]),
             "ent_type": np.array(outputs["entity"]),
             "sentence": tokens["sentence"]})


#####################################################
# External API functions below this line

def iob_to_spans(
    token_features: pd.DataFrame,
    iob_col_name: str = "ent_iob",
    char_span_col_name: str = "char_span",
    entity_type_col_name: str = "ent_type",
):
    """
    Convert token tags in Inside–Outside–Beginning (IOB) format to a series of
    `TokenSpan`s of entities.
    :param token_features: DataFrame of token features in the format returned by
     `make_tokens_and_features`.
    :param iob_col_name: Name of a column in `token_features` that contains the
     IOB tags as strings, "I", "O", or "B".
    :param char_span_col_name: Name of a column in `token_features` that
     contains the tokens as a `CharSpanArray`.
    :param entity_type_col_name: Optional name of a column in `token_features`
     that contains entity type information; or `None` if no such column exists.
    :return: A `pd.DataFrame` with the following columns:
    * `token_span`: Span (with token offsets) of each entity
    * `<value of entity_type_col_name>`: (optional) Entity type
    """
    # Start out with 1-token prefixes of all entities.
    begin_mask = token_features[iob_col_name] == "B"
    first_tokens = token_features[begin_mask].index
    if entity_type_col_name is None:
        entity_types = np.zeros(len(first_tokens))
    else:
        entity_types = token_features[begin_mask][entity_type_col_name]

    # Add an extra "O" tag to the end of the IOB column to simplify the logic
    # for handling the case where the document ends with an entity.
    iob_series = (
        token_features[iob_col_name].append(pd.Series(["O"])).reset_index(drop=True)
    )

    entity_prefixes = pd.DataFrame(
        {
            "ent_type": entity_types,
            "begin": first_tokens,  # Inclusive
            "end": first_tokens + 1,  # Exclusive
            "next_tag": iob_series.iloc[first_tokens + 1].values,
        }
    )

    df_list = []  # Type: pd.DataFrame

    if len(entity_prefixes.index) == 0:
        # Code below needs at least one element in the list for schema
        df_list = [entity_prefixes]

    # Iteratively expand the prefixes
    while len(entity_prefixes.index) > 0:
        complete_mask = entity_prefixes["next_tag"].isin(["O", "B"])
        complete_entities = entity_prefixes[complete_mask]
        incomplete_entities = entity_prefixes[~complete_mask].copy()
        incomplete_entities["end"] = incomplete_entities["end"] + 1
        incomplete_entities["next_tag"] = iob_series.iloc[
            incomplete_entities["end"]
        ].values
        df_list.append(complete_entities)
        entity_prefixes = incomplete_entities
    all_entities = pd.concat(df_list)

    # Sort spans by location, not length.
    all_entities.sort_values("begin", inplace=True)

    # Convert [begin, end) pairs to spans
    entity_spans_array = TokenSpanArray(
        token_features[char_span_col_name].values,
        all_entities["begin"].values,
        all_entities["end"].values,
    )
    if entity_type_col_name is None:
        return pd.DataFrame({"token_span": entity_spans_array})
    else:
        return pd.DataFrame(
            {
                "token_span": entity_spans_array,
                entity_type_col_name: all_entities["ent_type"].values,
            }
        )


def conll_2003_to_dataframes(input_file: str) -> List[pd.DataFrame]:
    """
    Parse a file in CoNLL-2003 training/test format into a DataFrame.

    CoNLL-2003 training/test format looks like this:
    ```
    SOCCER O
    - O
    JAPAN I-LOC
    GET O
    LUCKY O
    WIN O
    , O
    CHINA I-PER
    IN O
    SURPRISE O
    DEFEAT O
    . O

    Nadim I-PER
    Ladki I-PER

    AL-AIN I-LOC
    , O
    ```
    Note the presence of the surface forms of tokens at the beginning
    of the lines.

    :param input_file: Location of input file to read.
    :return: A list containing, for each document in the input file,
    a separate `pd.DataFrame` of four columns:
    * `char_span`: Span of each token, with character offsets.
      Backed by the concatenation of the tokens in the document into
      a single string with one sentence per line.
    * `token_span`: Span of each token, with token offsets.
      Backed by the contents of the `char_span` column.
    * `ent_iob`: IOB-format tags of tokens, corrected so that every
      entity begins with a "B" tag.
    * `ent_type`: Entity type names for tokens tagged "I" or "B" in
      the `ent_iob` column; `None` everywhere else.
    """
    return [_fix_iob_tags(_doc_to_df(d))
            for d in _parse_conll_file(input_file)]


def conll_2003_output_to_dataframes(doc_dfs: List[pd.DataFrame],
                                    input_file: str,
                                    copy_tokens: bool = False) -> List[pd.DataFrame]:
    """
    Parse a file in CoNLL-2003 output format into a DataFrame.

    CoNLL-2003 output format looks like this:
    ```
    O
    O
    I-LOC
    O
    O

    I-PER
    I-PER
    ```
    Note the lack of any information about the tokens themselves. Note
    also the lack of any information about document boundaries.

    :param doc_dfs: List of `pd.DataFrame`s of token information, as
     returned by `conll_2003_to_dataframes`. This is needed because
     CoNLL-2003 output format does not include any information about
     document boundaries.
    :param input_file: Location of input file to read.
    :param copy_tokens: If True, deep-copy token series from the
     elements of `doc_dfs` instead of using pointers.
    :return: A list containing, for each document in the input file,
    a separate `pd.DataFrame` of four columns:
    * `char_span`: Span of each token, with character offsets.
      Backed by the concatenation of the tokens in the document into
      a single string with one sentence per line.
    * `token_span`: Span of each token, with token offsets.
      Backed by the contents of the `char_span` column.
    * `ent_iob`: IOB-format tags of tokens, corrected so that every
      entity begins with a "B" tag.
    * `ent_type`: Entity type names for tokens tagged "I" or "B" in
      the `ent_iob` column; `None` everywhere else.
    """
    docs_list = _parse_conll_output_file(doc_dfs, input_file)
    return [
        _fix_iob_tags(_output_doc_to_df(tokens, outputs, copy_tokens))
        for tokens, outputs in zip(doc_dfs, docs_list)
    ]



