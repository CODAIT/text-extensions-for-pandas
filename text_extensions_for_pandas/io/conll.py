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

"""
The ``io.conll`` module contains I/O functions related to CoNLL-2003 file format and
its derivatives, including CoNLL-U.
"""

from typing import *

import numpy as np
import pandas as pd
import regex
import requests
import os
from zipfile import ZipFile

from text_extensions_for_pandas.array.span import SpanArray, SpanDtype
from text_extensions_for_pandas.array.token_span import (
    TokenSpan,
    TokenSpanArray,
)

# Special token that CoNLL-2003 format uses to delineate the documents in
# the collection.
_CONLL_DOC_SEPARATOR = "-DOCSTART-"
_EWT_DOC_SEPERATOR = "# newdoc id"

# _PUNCT_REGEX = regex.compile(f"[{string.punctuation}]+")
_PUNCT_OR_RIGHT_PAREN_REGEX = regex.compile(
    # Punctuation, right paren, or apostrophe followed by 1-2 lowercase letters
    # But not single or double quote, which could either begin or end a quotation
    "[!#%)*+,-./:;=>?@\\]^_`|}~]|'[a-zA-Z]{1,2}"
)
# Tokens that behave like left parentheses for whitespace purposes,
# including dollar signs ("$100", not "$ 100")
_LEFT_PAREN_REGEX = regex.compile(r"[(<\[{$]+")

# _PUNCT_MATCH_FN = np.vectorize(lambda s: _PUNCT_REGEX.fullmatch(s) is not None)
_SPACE_BEFORE_MATCH_FN = np.vectorize(
    lambda s: _PUNCT_OR_RIGHT_PAREN_REGEX.fullmatch(s) is not None
)
_SPACE_AFTER_MATCH_FN = np.vectorize(
    lambda s: _LEFT_PAREN_REGEX.fullmatch(s) is not None
)
_DEFAULT_CONLL_U_FORMAT = [
    "lemma",
    "upostag",
    "xpostag",
    "features",
    "head",
    "deprel",
    "deps",
    "misc",
]
_DEFAULT_CONLL_U_NUMERIC_COLS = ["head", "line_num"]

# What metadata to log from conllu (especially ewt) files.
# This is a dict as follows: tag_in_file -> desired name.
# When the tag in the file is seen in a comment, the following value will be stored
# and be assumed to apply to all elements in that document.
_DEFAULT_EWT_METADATA = {
    "sent_id": "sentence_id",
    "newpar id": "paragraph_id",
    "newdoc id": "doc_id",
}


# Note, Index in sentence is explicit; starts one further long
# for more information see https://universaldependencies.org/docs/format.html


def _make_empty_meta_values(
    column_names: List[str], iob_columns: List[bool]
) -> Dict[str, List[str]]:
    ret = {}
    for i in range(len(column_names)):
        name = column_names[i]
        if i >= len(iob_columns) or not iob_columns[i]:
            ret[name] = []
        else:
            ret[f"{name}_iob"] = []
            ret[f"{name}_type"] = []
    return ret


class _SentenceData:
    """
    Data structure that encapsulates one sentence's worth of data
    from a parsed CoNLL-2003 file.

    Not intended for use outside this file.
    """

    def __init__(
        self,
        column_names: List[str],
        iob_columns: List[bool],
        predicate_args: bool,
        conllu_metadata_cols: List[str] = None,
    ):
        self._column_names = column_names
        self._iob_columns = iob_columns
        self._num_standard_cols = len(self._column_names)

        # metadata-- init to None
        self._token_metadata = None

        # Surface form of token
        self._tokens = []  # Type: List[str]

        # Line numbers for each token from the file
        self._line_nums = []  # Type: List[int]

        # metadata from conll_u file
        self._conllu_metadata = (
            dict.fromkeys(conllu_metadata_cols, "")
            if conllu_metadata_cols is not None
            else None
        )
        self._conllu_metadata_exists = False

        self._conll_09_format = predicate_args

    @property
    def num_tokens(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    @property
    def token_metadata(self) -> Dict[str, List[str]]:
        return self._token_metadata

    @property
    def line_nums(self):
        return self._line_nums

    @property
    def column_names(self):
        return self._column_names

    @property
    def conll_u_metadata_feilds(self) -> List[str]:
        return (
            list(self._conllu_metadata.keys())
            if self._conllu_metadata is not None
            else None
        )

    @property
    def has_conll_u_metadata(self):
        return self._conllu_metadata_exists

    def set_conll_u_metadata(self, field: str, val: str):
        if str != "":
            self._conllu_metadata_exists = True
        self._conllu_metadata[field] = val
        self._update_conllu_metadata_exists()

    def set_batch_conll_u_metadata(self, metadata: Dict[str, str]):
        assert metadata.keys() <= self._conllu_metadata.keys()
        self._conllu_metadata.update(metadata)
        self._update_conllu_metadata_exists()

    def get_conll_u_metadata(self, field: str) -> str:
        return self._conllu_metadata[field]

    def _update_conllu_metadata_exists(self):
        self._conllu_metadata_exists = any(
            [v is not None and v != "" for v in self._conllu_metadata.values()]
        )

    def _process_line_tags(
        self,
        raw_tags: List[str],
        line_num: int,
        line_elems: List[str],
        is_conll_u: bool = False,
    ):
        if self._token_metadata is None:
            self._token_metadata = _make_empty_meta_values(
                self._column_names, self._iob_columns
            )

        for i in range(len(raw_tags)):
            raw_tag = raw_tags[i]
            name = self._column_names[i]
            if not self._iob_columns[i]:
                # non-IOB data
                self._token_metadata[name].append(raw_tag)
            else:
                # IOB-format data; split into two values
                if raw_tag.startswith("I-") or raw_tag.startswith("B-"):
                    # Tokens that are entities are tagged with tags like
                    # "I-PER" or "B-MISC".
                    tag, entity = raw_tag.split("-")
                elif raw_tag == "O":
                    tag = raw_tag
                    entity = None
                elif (not is_conll_u) and raw_tag == "-X-":
                    # Special metadata value for -DOCSTART- tags in the CoNLL corpus.
                    tag = "O"
                    entity = None
                else:
                    raise ValueError(
                        f"Tag '{raw_tag}' of IOB-format field {i} at line "
                        f"{line_num} does not start with 'I-', 'O', "
                        f"or 'B-'.\n"
                        f"Fields of line are: {line_elems}"
                    )
                self._token_metadata[f"{name}_iob"].append(tag)
                self._token_metadata[f"{name}_type"].append(entity)

    def add_line(self, line_num: int, line_elems: List[str]):
        """
        :param line_num: Location in file, for error reporting
        :param line_elems: Fields of a line, pre-split
        """
        if len(line_elems) != 1 + len(self._column_names):
            raise ValueError(
                f"Unexpected number of elements {len(line_elems)} "
                f"at line {line_num}; expected "
                f"{1 + len(self._column_names)} elements."
            )
        token = line_elems[0]
        raw_tags = line_elems[1:]
        self._tokens.append(token)
        self._line_nums.append(line_num)
        self._process_line_tags(raw_tags, line_num, line_elems, is_conll_u=False)

    def add_line_conllu(self, line_num: int, line_elems: List[str]):
        """
        Similar to add_line, but handles additional logic for conllu files.
        This includes the additional ignored entries on the left for word indexes within

        :param line_num: Location in file, for error reporting
        :param line_elems: Fields of a line, pre-split
        """
        if len(line_elems) < 2 + len(self._column_names):
            if len(line_elems) >= 2 + self._num_standard_cols:
                line_elems.extend(
                    ["_" for i in range(2 + len(self._column_names) - len(line_elems))]
                )
            else:
                raise ValueError(
                    f"Unexpected number of elements {len(line_elems)} "
                    f"at line {line_num}; expected "
                    f"{2 + len(self._column_names)} elements, "
                    f"got {len(line_elems)} instead."
                    f" min_num: {self._num_standard_cols}"
                    f"\nline reads: '{'    '.join(line_elems) }'"
                )
        if (
            len(line_elems) > 2 + len(self._column_names)
            and self._conll_09_format
            and self.num_tokens == 0
        ):
            # only modify once per sentence
            additional_lines = len(line_elems) - (3 + len(self._column_names))
            self._column_names.append("predicate")
            addnl_col_names = [f"pred{i}arg" for i in range(additional_lines)]
            self._column_names.extend(addnl_col_names)
            self._iob_columns.extend([False for i in range(additional_lines + 1)])
            # print(f"found Conll9 format. Added{additional_lines} columns. cols are now {self._column_names}")
            assert len(self._column_names) + 2 == len(line_elems)

        token = line_elems[1]
        raw_tags = line_elems[2 : len(self._column_names) + 2]
        raw_tags = [None if tag == "_" else tag for tag in raw_tags]
        self._tokens.append(token)
        self._line_nums.append(line_num)
        # because we do not combine
        self._process_line_tags(raw_tags, line_num, line_elems, is_conll_u=True)


def _parse_conll_file(
    input_file: str, column_names: List[str], iob_columns: List[bool]
) -> List[List[_SentenceData]]:
    """
    Parse the CoNLL-2003 file format for training/test data to Python
    objects.

    The format is especially tricky, so everything here is straight
    non-vectorized Python code. If you want performance, write the
    contents of your CoNLL files back out into a file format that
    supports performance.

    :param input_file: Location of the file to read
    :param column_names: Names for the metadata columns that come after the
     token text. These names will be used to generate the names of the dataframe
     that this function returns.
    :param iob_columns: Mask indicating which of the metadata columns after the
     token text should be treated as being in IOB format. If a column is in IOB format,
     the returned data structure will contain *two* columns, holding IOB tags and
     entity type tags, respectively. For example, an input column "ent" will turn into
     output columns "ent_iob" and "ent_type".

    :returns: A list of lists of _SentenceData objects. The top list has one entry per
     document. The next level lists have one entry per sentence.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Build up a list of document metadata as Python objects
    docs = []  # Type: List[List[Dict[str, List[str]]]]

    current_sentence = _SentenceData(column_names, iob_columns, False)

    # Information about the current document
    sentences = []  # Type: SentenceData

    for i in range(len(lines)):
        line = lines[i].strip()
        if 0 == len(line):
            # Blank line is the sentence separator
            if current_sentence.num_tokens > 0:
                sentences.append(current_sentence)
                current_sentence = _SentenceData(column_names, iob_columns, False)
        else:
            # Not at the end of a sentence
            line_elems = line.split(" ")
            current_sentence.add_line(i, line_elems)

            if line_elems[0] == _CONLL_DOC_SEPARATOR and i > 0:
                # End of document.  Wrap up this document and start a new one.
                #
                # Note that the special "start of document" token is considered part
                # of the document. If you do not follow this convention, the
                # result sets from CoNLL 2003 won't line up.
                # Note also that `current_sentence` is not in `sentences` and will be
                # added to the next document.
                docs.append(sentences)
                sentences = []

    # Close out the last sentence and document, if needed
    if current_sentence.num_tokens > 0:
        sentences.append(current_sentence)
    if len(sentences) > 0:
        docs.append(sentences)
    return docs


def _parse_conll_u_file(
    input_file: str,
    column_names: List[str],
    iob_columns: List[bool],
    predicate_args: bool = True,
    merge_subtokens: bool = False,
    merge_subtoken_separator: str = "|",
    metadata_fields: Dict[str, str] = _DEFAULT_EWT_METADATA,
    doc_seperator = _EWT_DOC_SEPERATOR
) -> List[List[_SentenceData]]:
    """



    The format is especially tricky, so everything here is straight
    non-vectorized Python code. If you want performance, write the
    contents of your CoNLL files back out into a file format that
    supports performance.

    :param input_file: Location of the file to read
    :param column_names: Names for the metadata columns that come after the
     token text. These names will be used to generate the names of the dataframe
     that this function returns.
    :param iob_columns: Mask indicating which of the metadata columns after the
     token text should be treated as being in IOB format. If a column is in IOB format,
     the returned data structure will contain *two* columns, holding IOB tags and
     entity type tags, respectively. For example, an input column "ent" will turn into
     output columns "ent_iob" and "ent_type".
    :param predicate_args: whether or not predicate arguments are stored in this file format.
    :param metadata_fields: the types of metadata fields you want to store from the docuement. in the form of a
    dictionary: tag_in_text -> "pretty" tag (i.e. what you want to show in the output)


    :returns: A list of lists of _SentenceData objects. The top list has one entry per
     document. The next level lists have one entry per sentence.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Build up a list of document metadata as Python objects
    docs = []  # Type: List[List[Dict[str, List[str]]]]

    # metadata specific to conll_u
    metadata_names = list(metadata_fields.values())
    u_metadata = dict.fromkeys(metadata_names, "")

    current_sentence = _SentenceData(
        column_names.copy(), iob_columns.copy(), predicate_args, metadata_names
    )

    # Information about the current document
    sentences = []  # Type: SentenceData

    # if we merge subtokens we need additional logic
    in_subtok = False  # set this flag when inside of subtoken
    subtok_end = None  # only valid when in subtok

    for i in range(len(lines)):
        line = lines[i].strip()
        if 0 == len(line):
            # Blank line is the sentence separator
            if current_sentence.num_tokens > 0:
                sentences.append(current_sentence)
                current_sentence = _SentenceData(
                    column_names.copy(),
                    iob_columns.copy(),
                    predicate_args,
                    metadata_names,
                )
                current_sentence.set_batch_conll_u_metadata(u_metadata)
        elif line[0] == "#":
            line_elems = line.split(" = ")
            if line_elems[0] == doc_seperator:
                if i > 0 and len(sentences) > 0 :
                    # End of document.  Wrap up this document and start a new one.
                    #
                    docs.append(sentences)
                    sentences = []
                    # reset doc, paragraph and sentence id's
            # now check for metadata
            line_elems[0] = line_elems[0].strip("# ")
            if line_elems[0] in metadata_fields.keys():
                key = metadata_fields[line_elems[0]]
                current_sentence.set_conll_u_metadata(key, line_elems[1])
                u_metadata[key] = line_elems[1]

        elif not in_subtok:
            # Not at the end of a sentence, or in a subtok
            line_elems = line.split("\t")
            # Ignore multi-word tokens for now; just use word sequence; may want to change, but we'd need to
            # interpret each sub-word's info

            if "-" not in line_elems[0]:  # checks if has range
                current_sentence.add_line_conllu(i, line_elems)
            elif merge_subtokens:
                in_subtok = True
                # find start and end of range
                start, end = line_elems[0].split("-")
                subtok_end = (
                    int(end) - int(start) + i + 1
                )  # the end (inclusive) of subtoken, by global index
                comb_elem_list = [[] for i in range(len(line_elems))]

                for subtoken in lines[i + 1 : subtok_end + 1]:
                    subtok_elems = subtoken.split("\t")
                    for field in range(2, len(line_elems)):
                        if subtok_elems[field] != "_":
                            comb_elem_list[field].append(subtok_elems[field])
                combined_elems = line_elems[0:2]  # first line is the same
                for elem_list in comb_elem_list[2:]:
                    combined_elems.append(merge_subtoken_separator.join(elem_list))

                current_sentence.add_line_conllu(i, combined_elems)

        if in_subtok and i >= subtok_end:
            in_subtok = False
            subtok_end = None

    # Close out the last sentence and document, if needed
    if current_sentence.num_tokens > 0:
        sentences.append(current_sentence)
    if len(sentences) > 0:
        docs.append(sentences)
    return docs


def _parse_conll_output_file(
    doc_dfs: List[pd.DataFrame], input_file: str
) -> List[Dict[str, List[str]]]:
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
     * `iob`: List of IOB2 tags as strings. This function does **NOT**
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
        line = lines[i].strip()
        if 0 == len(line):
            # Blank line is the sentence separator.
            continue
        if " " in line:
            raise ValueError(
                f"Line {i} contains unexpected space character.\n" f"Line was: '{line}'"
            )
        raw_tag = line
        if raw_tag.startswith("I") or raw_tag.startswith("B"):
            # Tokens that are entities are tagged with tags like
            # "I-PER" or "B-MISC".
            tag, entity = raw_tag.split("-")
        elif raw_tag == "O":
            tag = raw_tag
            entity = None
        else:
            raise ValueError(
                f"Unexpected tag {raw_tag} at line {i}.\n" f"Line was: '{line}'"
            )
        iobs.append(tag)
        entities.append(entity)
        token_num += 1
        if token_num == num_tokens_in_doc:
            # End of current document, advance to next
            docs.append({"iob": iobs, "entity": entities})
            iobs = []
            entities = []
            doc_num += 1
            token_num = 0
            if doc_num < len(doc_dfs):
                num_tokens_in_doc = len(doc_dfs[doc_num].index)

    if doc_num < len(doc_dfs):
        print(
            f"WARNING: Corpus has {len(doc_dfs)} documents, but "
            f"only found outputs for {doc_num} of them."
        )
        # raise ValueError(f"Corpus has {len(doc_dfs)} documents, but "
        #                  f"only found outputs for {doc_num} of them.")

    return docs


def _iob_to_iob2(
    df: pd.DataFrame, column_names: List[str], iob_columns: List[bool]
) -> pd.DataFrame:
    """
    In CoNLL-2003 format, entities are stored in IOB format, where the first
    token of an entity is only tagged "B" when there are two entities of the
    same type back-to-back. This format makes downstream processing difficult.
    If a given position has an `I` tag, that position may or may not be the
    first token of an entity. Code will need to inspect both the I/O/B tags
    *and* the entity type of multiple other tokens *and* the boundaries between
    sentences to disambiguate between those two cases.

    This function converts these IOB tags to the easier-to-consume IOB2 format;
    see
    https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
    for details. Basically, every entity in IOB2 format begins with a `B` tag.
    The `I` tag is only used for the second, third, etc. tokens of an entity.

    :param df: A `pd.DataFrame` with one row per token of the document.
     In addition to the metadata columns corresponding to `column_names`, this
     dataframe must also contain sentence information in a column called `sentence`.
    :param column_names: Names for the metadata columns in the original data file
     that were used to generate the names of the columns of `df`.
    :param iob_columns: Mask indicating which of the metadata columns after the
     token text should be treated as being in IOB format.

    :returns: A version of `df` with corrected IOB2 tags in the `ent_iob`
     column. The original dataframe is not modified.
    """
    ret = df.copy()
    sentence_begins = df["sentence"].values.begin_token

    for i in range(len(iob_columns)):
        if iob_columns[i]:
            name = column_names[i]
            iobs = df[f"{name}_iob"].values.copy()  # Modified in place
            entities = df[f"{name}_type"].values
            # Special-case the first one
            if iobs[0] == "I":
                iobs[0] = "B"
            for i in range(1, len(iobs)):
                tag = iobs[i]
                prev_tag = iobs[i - 1]
                if tag == "I":
                    if (
                        prev_tag == "O"  # Previous token not an entity
                        or (
                            prev_tag in ("I", "B") and entities[i] != entities[i - 1]
                        )  # Previous token a different type of entity
                        or (
                            sentence_begins[i] != sentence_begins[i - 1]
                        )  # Start of new sentence
                    ):
                        iobs[i] = "B"
            ret[f"{name}_iob"] = iobs
    return ret


def _doc_to_df(
    doc: List[_SentenceData],
    column_names: List[str],
    iob_columns: List[bool],
    space_before_punct: bool,
    conll_u: bool = False,
) -> pd.DataFrame:
    """
    Convert the "Python objects" representation of a document from a
    CoNLL-2003 file into a `pd.DataFrame` of token metadata.

    :param doc: List of Python objects that represents the document.
    :param column_names: Names for the metadata columns that come after the
     token text. These names will be used to generate the names of the dataframe
     that this function returns.
    :param iob_columns: Mask indicating which of the metadata columns after the
     token text should be treated as being in IOB format. If a column is in IOB format,
     the returned dataframe will contain *two* columns, holding IOB2 tags and
     entity type tags, respectively. For example, an input column "ent" will turn into
     output columns "ent_iob" and "ent_type".
    :param space_before_punct: If `True`, add whitespace before
     punctuation characters (and after left parentheses)
     when reconstructing the text of the document.
    :return: DataFrame with four columns:
    * `span`: Span of each token, with character offsets.
      Backed by the concatenation of the tokens in the document into
      a single string with one sentence per line.
    * `ent_iob`: IOB2-format tags of tokens, exactly as they appeared
      in the original file, with no corrections applied.
    * `ent_type`: Entity type names for tokens tagged "I" or "B" in
      the `ent_iob` column; `None` everywhere else.
    * `line_num`: line number of each token in the parsed file
    """

    # Character offsets of tokens in the reconstructed document
    begins_list = []  # Type: List[np.ndarray]
    ends_list = []  # Type: List[np.ndarray]

    # Reconstructed text of each sentence
    sentences_list = []  # Type: List[np.ndarray]

    # Token offsets of sentences containing each token in the document.
    sentence_begins_list = []  # Type: List[np.ndarray]
    sentence_ends_list = []  # Type: List[np.ndarray]

    # conll_u metadata information.
    conll_u_ids_exsist = doc is not None and len(doc)!=0 and doc[0].has_conll_u_metadata
    conll_2009_format  = doc is not None and len(doc)!=0 and doc[0]._conll_09_format
    # this should be the same for all sentences so we check the first

    if conll_2009_format:
        max_list = max(doc, key=lambda sent: len(sent.column_names)).column_names
        if len(max_list) > len(column_names):
            column_names = max_list

    # Token metadata column values. Key is column name, value is metadata for
    # each token.
    if conll_u_ids_exsist:
        meta_lists = _make_empty_meta_values(
            column_names + doc[0].conll_u_metadata_feilds, iob_columns
        )
    else:
        meta_lists = _make_empty_meta_values(column_names, iob_columns)

    # Line numbers of the parsed file for each token in the doc
    doc_line_nums = []

    char_position = 0
    token_position = 0
    for sentence_num in range(len(doc)):
        sentence = doc[sentence_num]
        tokens = sentence.tokens

        # Don't put spaces before punctuation in the reconstituted string.
        no_space_before_mask = (
            np.zeros(len(tokens), dtype=bool)
            if space_before_punct
            else _SPACE_BEFORE_MATCH_FN(tokens)
        )
        no_space_after_mask = (
            np.zeros(len(tokens), dtype=bool)
            if space_before_punct
            else _SPACE_AFTER_MATCH_FN(tokens)
        )
        no_space_before_mask[0] = True  # No space before first token
        no_space_after_mask[-1] = True  # No space after last token
        shifted_no_space_after_mask = np.roll(no_space_after_mask, 1)
        prefixes = np.where(
            np.logical_or(no_space_before_mask, shifted_no_space_after_mask), "", " "
        )
        string_parts = np.ravel((prefixes, tokens), order="F")
        sentence_text = "".join(string_parts)
        sentences_list.append(sentence_text)

        lengths = np.array([len(t) for t in tokens])
        prefix_lengths = np.array([len(p) for p in prefixes])

        # Begin and end offsets, accounting for which tokens have spaces
        # before them.
        e = np.cumsum(lengths + prefix_lengths)
        b = e - lengths
        begins_list.append(b + char_position)
        ends_list.append(e + char_position)

        sentence_begin_token = token_position
        sentence_end_token = token_position + len(e)
        sentence_begins = np.repeat(sentence_begin_token, len(e))
        sentence_ends = np.repeat(sentence_end_token, len(e))
        sentence_begins_list.append(sentence_begins)
        sentence_ends_list.append(sentence_ends)

        for k in meta_lists.keys():
            if k in sentence.token_metadata.keys():
                meta_lists[k].extend(sentence.token_metadata[k])
            elif conll_u_ids_exsist and k in sentence.conll_u_metadata_feilds:
                data = sentence.get_conll_u_metadata(k)
                meta_lists[k].extend([data for i in range(sentence.num_tokens)])
            else:
                meta_lists[k].extend([None for i in range(sentence.num_tokens)])

        char_position += e[-1] + 1  # "+ 1" to account for newline
        token_position += len(e)

        doc_line_nums.extend(sentence.line_nums)
        # move "head" indices so they point at the right words
        if conll_u and "head" in column_names:
            for i in range(sentence_begin_token, sentence_end_token):
                val = meta_lists["head"][i]
                if val is not None:
                    points_to = int(val)
                    meta_lists["head"][i] = (
                        points_to + sentence_begin_token - 1 if points_to != 0 else -1
                    )

    begins = np.concatenate(begins_list)
    ends = np.concatenate(ends_list)
    doc_text = "\n".join(sentences_list)
    char_spans = SpanArray(doc_text, begins, ends)
    sentence_spans = TokenSpanArray(
        char_spans,
        np.concatenate(sentence_begins_list),
        np.concatenate(sentence_ends_list),
    )

    ret = pd.DataFrame({"span": char_spans})
    for k, v in meta_lists.items():
        ret[k] = v
    ret["sentence"] = sentence_spans
    ret["line_num"] = pd.Series(doc_line_nums)
    if conll_u and "head" in column_names:
        ret = ret.astype({"head": "Int64"}, errors="ignore")
        ret.loc[ret["head"] == -1, "head"] = pd.NA
    return ret


def _output_doc_to_df(
    tokens: pd.DataFrame,
    outputs: Dict[str, List[str]],
    column_name: str,
    copy_tokens: bool,
) -> pd.DataFrame:
    """
    Convert the "Python objects" representation of a document from a
    CoNLL-2003 file into a `pd.DataFrame` of token metadata.

    :param tokens: `pd.DataFrame` containing metadata about the tokens
     of this document, as returned by `conll_2003_to_dataframe`
    :param outputs: Dictionary containing outputs for this document,
     with fields "iob" and "entity".
    :param column_name: Name for the metadata value that the IOB-tagged data
     in `input_file` encodes. If this name is present in `doc_dfs`, its value
     will be replaced with the data from `input_file`; otherwise a new column
     will be added to each dataframe.
    :param copy_tokens: `True` if token information should be deep-copied.
    :return: DataFrame with four columns:
    * `span`: Span of each token, with character offsets.
      Backed by the concatenation of the tokens in the document into
      a single string with one sentence per line.
    * `ent_iob`: IOB2-format tags of tokens, corrected so that every
      entity begins with a "B" tag.
    * `ent_type`: Entity type names for tokens tagged "I" or "B" in
      the `ent_iob` column; `None` everywhere else.
    """
    if copy_tokens:
        return pd.DataFrame(
            {
                "span": tokens["span"].copy(),
                f"{column_name}_iob": np.array(outputs["iob"]),
                f"{column_name}_type": np.array(outputs["entity"]),
                "sentence": tokens["sentence"].copy(),
            }
        )
    else:
        return pd.DataFrame(
            {
                "span": tokens["span"],
                f"{column_name}_iob": np.array(outputs["iob"]),
                f"{column_name}_type": np.array(outputs["entity"]),
                "sentence": tokens["sentence"],
            }
        )


#####################################################
# External API functions below this line


def iob_to_spans(
    token_features: pd.DataFrame,
    iob_col_name: str = "ent_iob",
    span_col_name: str = "span",
    entity_type_col_name: str = "ent_type",
):
    """
    Convert token tags in Inside–Outside–Beginning (IOB2) format to a series of
    :class:`TokenSpan` objects of entities. See See wikipedia_ for more information
    on the IOB2 format.

    .. _wikipedia: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

    :param token_features: DataFrame of token features in the format returned by
     :func:`make_tokens_and_features`.
    :param iob_col_name: Name of a column in ``token_features`` that contains the
     IOB2 tags as strings, "I", "O", or "B".
    :param span_col_name: Name of a column in ``token_features`` that
     contains the tokens as a :class:`SpanArray`.
    :param entity_type_col_name: Optional name of a column in ``token_features``
     that contains entity type information; or ``None`` if no such column exists.

    :returns: A :class:`pd.DataFrame` with the following columns:

        * ``span``: Span (with token offsets) of each entity
        * ``<value of entity_type_col_name>``: (optional) Entity type
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
        token_features[span_col_name].values,
        all_entities["begin"].values,
        all_entities["end"].values,
    )
    if entity_type_col_name is None:
        return pd.DataFrame({"span": entity_spans_array})
    else:
        return pd.DataFrame(
            {
                "span": entity_spans_array,
                entity_type_col_name: all_entities["ent_type"].values,
            }
        )


def spans_to_iob(
    token_spans: Union[TokenSpanArray, List[TokenSpan], pd.Series],
    span_ent_types: Union[str, Iterable, np.ndarray, pd.Series] = None,
) -> pd.DataFrame:
    """
    Convert a series of :class:`TokenSpan` objects of entities to token tags in
    Inside–Outside–Beginning (IOB2) format. See wikipedia_ for more information
    on the IOB2 format.

    .. _wikipedia: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

    :param token_spans: An object that can be converted to a :class:`TokenSpanArray` via
        :func:`TokenSpanArray.make_array`. Should contain :class:`TokenSpan` objects
        aligned with the target tokenization. All spans must be from the same document.
        Usually you create this array by calling :func:`TokenSpanArray.align_to_tokens`.
    :param span_ent_types: List of entity type strings corresponding to each of the
        elements of ``token_spans``, or ``None`` to indicate null entity tags.

    :returns: A :class:`pd.DataFrame` with two columns:

      * "ent_iob": IOB2 tags as strings "ent_iob"
      * "ent_type": Entity type strings (or NaN values if ``ent_types`` is ``None``)
    """
    # Normalize inputs
    token_spans = TokenSpanArray.make_array(token_spans)
    if span_ent_types is None:
        span_ent_types = [None] * len(token_spans)
    elif isinstance(span_ent_types, str):
        span_ent_types = [span_ent_types] * len(token_spans)
    elif isinstance(span_ent_types, pd.Series):
        span_ent_types = span_ent_types.values

    # Define the IOB categorical type with "O" == 0, "B"==1, "I"==2
    iob2_dtype = pd.CategoricalDtype(["O", "B", "I"], ordered=False)

    # Handle an empty token span array
    if len(token_spans) == 0:
        return pd.DataFrame(
            {
                "ent_iob": pd.Series(dtype=iob2_dtype),
                "ent_type": pd.Series(dtype="string"),
            }
        )

    # All code that follows assumes at least one input span. All spans should
    # be from the same document; otherwise there isn't a meaningful IOB
    # representation of the entities.
    if not token_spans.is_single_tokenization:
        raise ValueError(
            f"All input spans must be from the same tokenization of "
            f"the same document "
            f"(spans are {token_spans})"
        )

    tokens = token_spans.tokens[0]

    # Initialize an IOB series with all 'O' entities
    iob_data = np.zeros_like(tokens.begin, dtype=np.int64)
    iob_tags = pd.Categorical.from_codes(codes=iob_data, dtype=iob2_dtype)

    # Assign the begin tags
    iob_tags[token_spans.begin_token] = "B"

    # Fill in the remaining inside tags
    i_lengths = token_spans.end_token - (token_spans.begin_token + 1)
    i_mask = i_lengths > 0
    i_begins = token_spans.begin_token[i_mask] + 1
    i_ends = token_spans.end_token[i_mask]
    for begin, end in zip(i_begins, i_ends):
        iob_tags[begin:end] = "I"

    # Use a similar process to generate entity type tags
    ent_types = np.full(len(tokens), None, dtype=object)
    for ent_type, begin, end in zip(
        span_ent_types, token_spans.begin_token, token_spans.end_token
    ):
        ent_types[begin:end] = ent_type

    return pd.DataFrame(
        {"ent_iob": iob_tags, "ent_type": pd.Series(ent_types, dtype="string")}
    )


def conll_2003_to_dataframes(
    input_file: str,
    column_names: List[str],
    iob_columns: List[bool],
    space_before_punct: bool = False,
) -> List[pd.DataFrame]:
    """
    Parse a file in CoNLL-2003 training/test format into a DataFrame.

    CoNLL-2003 training/test format looks like this::

        -DOCSTART- -X- -X- O

        CRICKET NNP I-NP O
        - : O O
        LEICESTERSHIRE NNP I-NP I-ORG
        TAKE NNP I-NP O
        OVER IN I-PP O
        AT NNP I-NP O

    Note the presence of the surface forms of tokens at the beginning
    of the lines.

    :param input_file: Location of input file to read.
    :param space_before_punct: If ``True``, add whitespace before
     punctuation characters when reconstructing the text of the document.
    :param column_names: Names for the metadata columns that come after the
     token text. These names will be used to generate the names of the dataframe
     that this function returns.
    :param iob_columns: Mask indicating which of the metadata columns after the
     token text should be treated as being in IOB format. If a column is in IOB format,
     the returned dataframe will contain *two* columns, holding **IOB2** tags and
     entity type tags, respectively. For example, an input column "ent" will turn into
     output columns "ent_iob" and "ent_type".

    :returns: A list containing, for each document in the input file,
     a separate :class:`pd.DataFrame` of four columns:

        * **span**: Span of each token, with character offsets.
          Backed by the concatenation of the tokens in the document into
          a single string with one sentence per line.
        * **ent_iob** IOB2-format tags of tokens, corrected so that every
          entity begins with a "B" tag.
        * **ent_type**: Entity type names for tokens tagged "I" or "B" in
          the `ent_iob` column; `None` everywhere else.
    """
    parsed_docs = _parse_conll_file(input_file, column_names, iob_columns)
    doc_dfs = [
        _doc_to_df(d, column_names, iob_columns, space_before_punct)
        for d in parsed_docs
    ]
    return [_iob_to_iob2(d, column_names, iob_columns) for d in doc_dfs]


def conll_u_to_dataframes(
    input_file: str,
    column_names: List[str] = _DEFAULT_CONLL_U_FORMAT,
    iob_columns: List[bool] = None,
    has_predicate_args: bool = True,
    space_before_punct: bool = False,
    merge_subtokens: bool = False,
    merge_subtoken_separator: str = "|",
    numeric_cols: List[str] = _DEFAULT_CONLL_U_NUMERIC_COLS,
    metadata_fields: Dict[str, str] = _DEFAULT_EWT_METADATA,
    separate_sentences_by_doc = False
) -> List[pd.DataFrame]:
    """
    Parses a file from

    :param input_file: Location of input file to read.
    :param space_before_punct: If `True`, add whitespace before
     punctuation characters when reconstructing the text of the document.
    :param column_names: Names for the metadata columns that come after the
     token text. These names will be used to generate the names of the dataframe
     that this function returns. These default to the format defined at
     https://universaldependencies.org/docs/format.html, but can be modified if needed
    :param iob_columns: Mask indicating which of the metadata columns after the
     token text should be treated as being in IOB format. If a column is in IOB format,
     the returned dataframe will contain *two* columns, holding **IOB2** tags and
     entity type tags, respectively. For example, an input column "ent" will turn into
     output columns "ent_iob" and "ent_type". By default in CONLL_U or EWT formats this is all false.
    :param has_predicate_args: Whether or not the file format includes predicate args. True by default, and
     should support most files in the conllu format, but will assume that any tabs in the last
     element are additional predicate arguments
    :param merge_subtokens: dictates how to handle tokens that are smaller than one word. By default, we keep
     the subtokens as two seperate entities, but if this is set to true, the subtokens will be merged into a
     single entity, of the same length as the token, and their attributes will be concatenated
    :param merge_subtoken_separator: If merge subtokens is selected, concatenate the attributes with this
     separator, by default '|'
    :param metadata_fields: the types of metadata fields you want to store from the docuement. in the form of a
     dictionary: tag_in_text -> "pretty" tag (i.e. what you want to show in the output)

    :returns: A list containing, for each document in the input file,
     a separate `pd.DataFrame` of four columns:

        * `span`: Span of each token, with character offsets.
          Backed by the concatenation of the tokens in the document into
          a single string with one sentence per line.
        * `ent_iob`: IOB2-format tags of tokens, corrected so that every
          entity begins with a "B" tag.
        * `ent_type`: Entity type names for tokens tagged "I" or "B" in
          the `ent_iob` column; `None` everywhere else.

    """
    if iob_columns is None:
        iob_columns = [False for i in range(len(column_names))]
        # fill with falses if not specified
    
    # 
    split_doc_by = "# text" if separate_sentences_by_doc else _EWT_DOC_SEPERATOR

    parsed_docs = _parse_conll_u_file(
        input_file,
        column_names,
        iob_columns,
        has_predicate_args,
        merge_subtokens=merge_subtokens,
        merge_subtoken_separator=merge_subtoken_separator,
        metadata_fields=metadata_fields,
        doc_seperator=split_doc_by
    )
    doc_dfs = [
        _doc_to_df(d, column_names, iob_columns, space_before_punct, conll_u=True)
        for d in parsed_docs
    ]
    ret = [_iob_to_iob2(d, column_names, iob_columns) for d in doc_dfs]
    for d in ret:
        for col in numeric_cols:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return ret


def conll_2003_output_to_dataframes(
    doc_dfs: List[pd.DataFrame],
    input_file: str,
    column_name: str = "ent",
    copy_tokens: bool = False,
) -> List[pd.DataFrame]:
    """
    Parse a file in CoNLL-2003 output format into a DataFrame.

    CoNLL-2003 output format looks like this::
        O
        O
        I-LOC
        O
        O

        I-PER
        I-PER

    Note the lack of any information about the tokens themselves. Note
    also the lack of any information about document boundaries.

    :param doc_dfs: List of :class:`pd.DataFrame`s of token information, as
     returned by :func:`conll_2003_to_dataframes`. This is needed because
     CoNLL-2003 output format does not include any information about
     document boundaries.
    :param input_file: Location of input file to read.
    :param column_name: Name for the metadata value that the IOB-tagged data
     in ``input_file`` encodes. If this name is present in ``doc_dfs``, its value
     will be replaced with the data from ``input_file``; otherwise a new column
     will be added to each dataframe.
    :param copy_tokens: If ``True``, deep-copy token series from the
     elements of `doc_dfs` instead of using pointers.

    :returns: A list containing, for each document in the input file,
        a separate :class:`pd.DataFrame` of four columns:

        * **span**: Span of each token, with character offsets.
          Backed by the concatenation of the tokens in the document into
          a single string with one sentence per line.
        * **token_span**: Span of each token, with token offsets.
          Backed by the contents of the `span` column.
        * **<column_name>_iob**: IOB2-format tags of tokens, corrected so that every
          entity begins with a "B" tag.
        * **<column_name>_type**: Entity type names for tokens tagged "I" or "B" in
          the ``<column_name>_iob`` column; ``None`` everywhere else.
    """
    docs_list = _parse_conll_output_file(doc_dfs, input_file)

    return [
        _iob_to_iob2(
            _output_doc_to_df(tokens, outputs, column_name, copy_tokens),
            [column_name],
            [True],
        )
        for tokens, outputs in zip(doc_dfs, docs_list)
    ]


def make_iob_tag_categories(
    entity_types: List[str],
) -> Tuple[pd.CategoricalDtype, List[str], Dict[str, int]]:
    """
    Enumerate all the possible token categories for combinations of
    IOB tags and entity types (for example, ``I + "PER" ==> "I-PER"``).
    Generate a consistent mapping from these strings to integers.

    :param entity_types: Allowable entity type strings for the corpus

    :returns: A triple of:

     * Pandas CategoricalDtype
     * mapping from integer to string label, as a list. This mapping is guaranteed
       to be consistent with the mapping in the Pandas CategoricalDtype in the first
       return value.
     * mapping string label to integer, as a dict; the inverse of the second return
       value.
    """
    int_to_label = ["O"] + [f"{x}-{y}" for x in ["B", "I"] for y in entity_types]
    label_to_int = {int_to_label[i]: i for i in range(len(int_to_label))}
    token_class_dtype = pd.CategoricalDtype(categories=int_to_label)
    return token_class_dtype, int_to_label, label_to_int


def add_token_classes(
    token_features: pd.DataFrame,
    token_class_dtype: pd.CategoricalDtype = None,
    iob_col_name: str = "ent_iob",
    entity_type_col_name: str = "ent_type",
) -> pd.DataFrame:
    """
    Add additional columns to a dataframe of IOB-tagged tokens containing composite
    string and integer category labels for the tokens.

    :param token_features: Dataframe of tokens with IOB tags and entity type strings
    :param token_class_dtype: Optional Pandas categorical dtype indicating how to map
     composite tags like `I-PER` to integer values.
     You can use :func:`make_iob_tag_categories` to generate this dtype.
     If this parameter is not provided, this function will use an arbitrary mapping
     using the values that appear in this dataframe.
    :param iob_col_name: Optional name of a column in `token_features` that contains the
     IOB2 tags as strings, "I", "O", or "B".
    :param entity_type_col_name: Optional name of a column in `token_features`
     that contains entity type information; or `None` if no such column exists.

    :returns: A copy of `token_features` with two additional columns, `token_class`
     (string class label) and `token_class_id` (integer label).
     If `token_features` contains columns with either of these names, those columns will
     be overwritten in the returned copy of `token_features`.
    """
    if token_class_dtype is None:
        empty_mask = token_features[entity_type_col_name].isna() | (
            token_features[entity_type_col_name] == ""
        )
        token_class_type, _, label_to_int = make_iob_tag_categories(
            list(token_features[~empty_mask][entity_type_col_name].unique())
        )
    else:
        label_to_int = {
            token_class_dtype.categories[i]: i
            for i in range(len(token_class_dtype.categories))
        }
    elems = []  # Type: str
    for index, row in token_features[[iob_col_name, entity_type_col_name]].iterrows():
        if row[iob_col_name] == "O":
            elems.append("O")
        else:
            elems.append(f"{row[iob_col_name]}-{row[entity_type_col_name]}")
    ret = token_features.copy()
    ret["token_class"] = pd.Categorical(elems, dtype=token_class_dtype)
    ret["token_class_id"] = [label_to_int[l] for l in elems]
    return ret


def decode_class_labels(class_labels: Iterable[str]):
    """
    Decode the composite labels that :func:`add_token_classes` creates.

    :param class_labels: Iterable of string class labels like "I-LOC"
    :returns: A tuple of (IOB2 tags, entity type strings) corresponding
     to the class labels.
    """
    iobs = ["O" if t == "O" else t[:1] for t in class_labels]
    types = [None if t == "O" else t.split("-")[1] for t in class_labels]
    return iobs, types


def maybe_download_conll_data(target_dir: str) -> Dict[str, str]:
    """
    Download and cache a copy of the CoNLL-2003 named entity recognition
    data set.

    **NOTE: This data set is licensed for research use only.**
    Be sure to adhere to the terms of the license when using this data set!

    :param target_dir: Directory where this function should write the corpus
     files, if they are not already present.

    :returns: Dictionary containing a mapping from fold name to file name for
     each of the three folds (`train`, `test`, `dev`) of the corpus.
    """
    _CONLL_DOWNLOAD_BASE_URL = (
        "https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/"
        "data/conll2003/"
    )
    _TRAIN_FILE_NAME = "eng.train"
    _DEV_FILE_NAME = "eng.testa"
    _TEST_FILE_NAME = "eng.testb"
    _TRAIN_FILE = f"{target_dir}/{_TRAIN_FILE_NAME}"
    _DEV_FILE = f"{target_dir}/{_DEV_FILE_NAME}"
    _TEST_FILE = f"{target_dir}/{_TEST_FILE_NAME}"

    def download_file(url, destination):
        data = requests.get(url)
        open(destination, "wb").write(data.content)

    if not os.path.exists(_TRAIN_FILE):
        download_file(_CONLL_DOWNLOAD_BASE_URL + _TRAIN_FILE_NAME, _TRAIN_FILE)
    if not os.path.exists(_DEV_FILE):
        download_file(_CONLL_DOWNLOAD_BASE_URL + _DEV_FILE_NAME, _DEV_FILE)
    if not os.path.exists(_TEST_FILE):
        download_file(_CONLL_DOWNLOAD_BASE_URL + _TEST_FILE_NAME, _TEST_FILE)
    return {"train": _TRAIN_FILE, "dev": _DEV_FILE, "test": _TEST_FILE}


def maybe_download_dataset_data(
    target_dir: str, document_url: str, fname: str = None
) -> Union[str, List[str]]:
    """
     If the file found at the url is not found in the target directory,
     downloads it, and saves it to that place in downloads.
     Returns the path to the file. If a zip archive is downloaded, only files that are not already in the target
     directory will be fetched, and if an alternate_name is given only that file will be operated on.
     Note if a Zip archive is downloaded it will be unpacked so verify that the url being used is safe.

    :param target_dir: Directory where this function should write the document
    :param document_url: url from which to download the docuemnt. If no alternate name is specified,
     it is assumed that the string after the last slash is the name of the file.
    :param fname: if given, the name of the file that is checked in the target directory,
     as well as what is used to save the file if no such file is found. If a zip file is downloaded, and a file of this
     name exists in in the archive, only it will be extracted.

    :returns: the path to the file, or None if downloading was not successful
      If the file found at the url is not found in the target directory,
      downloads it, and saves it to that place in downloads
    """
    file_name = (
        fname if fname is not None else document_url.split("/")[-1]
    )
    full_path = target_dir + "/" + file_name
    # if no directory exists, create one
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # special logic for zip files
    if document_url.split(".")[-1] == "zip" and (
        fname is None or not os.path.exists(full_path)
    ):
        # if we have a zip file already, don't re-download it
        zipPath = target_dir + "/" + document_url.split("/")[-1]
        if not os.path.exists(zipPath):
            data = requests.get(document_url)
            open(zipPath, "wb").write(data.content)

        # if need be, extract the zipfile documents
        with ZipFile(zipPath, "r") as zipf:
            fnames = zipf.namelist()
            if fname is not None and fname in fnames:
                zipf.extract(fname, target_dir)
                return full_path
            for fname in fnames:
                if not os.path.exists(target_dir + fname):
                    zipf.extract(fname, target_dir)
        if len(fnames) == 1:
            full_path = target_dir + "/" + fnames[0]
        else:
            return [target_dir + "/" + fname for fname in fnames]

    # regular logic
    elif not os.path.exists(full_path):
        try:
            data = requests.get(document_url)
            open(full_path, "wb").write(data.content)
        except:
            return None
    return full_path


def _prep_for_stacking(fold_name: str, doc_num: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Subroutine of combine_folds()
    """
    df_values = {
        "fold": fold_name,
        "doc_num": doc_num,
    }
    for colname in df.columns:
        df_values[colname] = df[colname]
    return pd.DataFrame(df_values)


def combine_folds(fold_to_docs: Dict[str, List[pd.DataFrame]]):
    """
    Merge together multiple parts of a corpus (i.e. train, test, validation)
    into a single DataFrame of all tokens in the corpus.

    :param fold_to_docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`util.conll_to_bert`.
     All DataFrames must have the same schema, but any schema is ok.

    :returns: corpus wide DataFrame with some additional leading columns `fold`
     and `doc_num` to tell what fold and document number within the fold each
     row of the dataframe comes from.
    """
    to_stack = []  # Type: List[pd.DataFrame]
    for fold_name, docs_in_fold in fold_to_docs.items():
        to_stack.extend(
            [
                _prep_for_stacking(fold_name, i, docs_in_fold[i])
                for i in range(len(docs_in_fold))
            ]
        )
    return pd.concat(to_stack).reset_index(drop=True)


def compute_accuracy_by_document(
    corpus_dfs: Dict[Tuple[str, int], pd.DataFrame],
    output_dfs: Dict[Tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 scores by document.

    :param corpus_dfs: Gold-standard span/entity type pairs, as either:

     * a dictionary of DataFrames, one DataFrames per document, indexed by
       tuples of (collection name, offset into collection)
     * a list of DataFrames, one per document
       as returned by :func:`conll_2003_output_to_dataframes()`
    :param output_dfs: Model outputs, in the same format as `gold_dfs`
        (i.e. exactly the same column names). This is the format that
        produces.
    """
    if isinstance(corpus_dfs, list):
        if not isinstance(output_dfs, list):
            raise TypeError(
                f"corpus_dfs is a list, but output_dfs is of type "
                f"'{type(output_dfs)}', which is not a list."
            )
        corpus_dfs = {("", i): corpus_dfs[i] for i in range(len(corpus_dfs))}
        output_dfs = {("", i): output_dfs[i] for i in range(len(output_dfs))}
    # Note that it's important for all of these lists to be in the same
    # order; hence these expressions all iterate over gold_dfs.keys()
    num_true_positives = [
        len(corpus_dfs[k].merge(output_dfs[k]).index) for k in corpus_dfs.keys()
    ]
    num_extracted = [len(output_dfs[k].index) for k in corpus_dfs.keys()]
    num_entities = [len(corpus_dfs[k].index) for k in corpus_dfs.keys()]
    collection_name = [t[0] for t in corpus_dfs.keys()]
    doc_num = [t[1] for t in corpus_dfs.keys()]

    stats_by_doc = pd.DataFrame(
        {
            "fold": collection_name,
            "doc_num": doc_num,
            "num_true_positives": num_true_positives,
            "num_extracted": num_extracted,
            "num_entities": num_entities,
        }
    )
    stats_by_doc["precision"] = (
        stats_by_doc["num_true_positives"] / stats_by_doc["num_extracted"]
    )
    stats_by_doc["recall"] = (
        stats_by_doc["num_true_positives"] / stats_by_doc["num_entities"]
    )
    stats_by_doc["F1"] = (
        2.0
        * (stats_by_doc["precision"] * stats_by_doc["recall"])
        / (stats_by_doc["precision"] + stats_by_doc["recall"])
    )
    return stats_by_doc


def compute_global_accuracy(stats_by_doc: pd.DataFrame):
    """
    Compute collection-wide precision, recall, and F1 score from the
    output of :func:`compute_f1_by_document`.

    :param stats_by_doc: Output of :func:`make_stats_df`
    :returns: A Python dictionary of collection-level statistics about
     result quality.
    """
    num_true_positives = stats_by_doc["num_true_positives"].sum()
    num_entities = stats_by_doc["num_entities"].sum()
    num_extracted = stats_by_doc["num_extracted"].sum()

    precision = num_true_positives / num_extracted
    recall = num_true_positives / num_entities
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return {
        "num_true_positives": num_true_positives,
        "num_entities": num_entities,
        "num_extracted": num_extracted,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }
