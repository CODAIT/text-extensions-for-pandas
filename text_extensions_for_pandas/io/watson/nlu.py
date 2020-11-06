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
This module of Text Extensions for Pandas includes I/O functions related to the
Watson Natural Language Understanding service on the IBM Cloud.
This service provides analysis of text feature through a request/response API.
See
https://cloud.ibm.com/docs/natural-language-understanding?topic=natural-language-understanding-getting-started
for information on getting started with the service. Details of the API and
available features can be found at
https://cloud.ibm.com/apidocs/natural-language-understanding?code=python#introduction.
For convenience, a Python SDK is available at
https://github.com/watson-developer-cloud/python-sdk
that can be used to authenticate and make requests to the service.
"""

from typing import *
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from text_extensions_for_pandas.array.span import SpanArray, Span
from text_extensions_for_pandas.array.token_span import TokenSpanArray
from text_extensions_for_pandas.io.watson import util
from text_extensions_for_pandas.spanner import contain_join


# Standard Schemas for Response Data
_entities_schema = [
    ("type", "string"),
    ("text", "string"),
    ("sentiment.label", "string"),
    ("sentiment.score", "double"),
    ("relevance", "double"),
    ("count", "int64"),
    ("confidence", "double"),
    ("disambiguation.subtype", "string"),
    ("disambiguation.name", "string"),
    ("disambiguation.dbpedia_resource", "string"),
]

# Schema for a DataFrame that unrolls the nested field "mentions" containing
# the locations of individual mentions of an entity.
# Fields from parent "entities" relation:
_entity_mentions_parent_elems = [
    ("type", "string"),
    ("text", "string"),
    # Sentiment is not an attribute of the individual mentions
    # ("sentiment.label", "string"),
    # ("sentiment.score", "double"),
    # ("relevance", "double"),
    # ("count", "int64"),
    # ("confidence", "double"),
    # ("disambiguation.subtype", "string"),
    # ("disambiguation.name", "string"),
    # ("disambiguation.dbpedia_resource", "string"),
]
# Fields from nested "mentions" field that we unroll:
_entity_mentions_child_elems = [
    ("span", "ArrowSpanType"),  # NOTE: Renamed from "location"
    ("confidence", "double")
]
_entity_mentions_schema = _entity_mentions_parent_elems + _entity_mentions_child_elems
_entity_mentions_parent_names = util.schema_to_names(_entity_mentions_parent_elems)
_entity_mentions_child_names = util.schema_to_names(_entity_mentions_child_elems)

_keywords_schema = [
    ("text", "string"),
    ("sentiment.label", "string"),
    ("sentiment.score", "double"),
    ("relevance", "double"),
    ("emotion.sadness", "double"),
    ("emotion.joy", "double"),
    ("emotion.fear", "double"),
    ("emotion.disgust", "double"),
    ("emotion.anger", "double"),
    ("count", "int64"),
]

_semantic_roles_schema = [
    ("subject.text", "string"),
    ("sentence", "string"),
    ("object.text", "string"),
    ("action.verb.text", "string"),
    ("action.verb.tense", "string"),
    ("action.text", "string"),
    ("action.normalized", "string"),
]

_syntax_schema = [
    ("span", "ArrowSpanType"),
    ("part_of_speech", "string"),
    ("lemma", "string"),
    ("sentence", " ArrowTokenSpanType"),
]

_relations_schema = [
    ("type", "string"),
    ("sentence_span", "ArrowTokenSpanType"),
    ("score", "double"),
    ("arguments.0.span", "ArrowTokenSpanType"),
    ("arguments.1.span", "ArrowTokenSpanType"),
    ("arguments.0.entities.type", "string"),
    ("arguments.1.entities.type", "string"),
    ("arguments.0.entities.text", "string"),
    ("arguments.1.entities.text", "string"),
    ("arguments.0.entities.disambiguation.subtype", "string"),
    ("arguments.1.entities.disambiguation.subtype", "string"),
    ("arguments.0.disambiguation.name", "string"),
    ("arguments.1.disambiguation.name", "string"),
    ("arguments.0.disambiguation.dbpedia_resource", "string"),
    ("arguments.1.disambiguation.dbpedia_resource", "string"),
]


def _make_syntax_dataframes(syntax_response, original_text):
    tokens = syntax_response.get("tokens", [])
    sentence = syntax_response.get("sentences", [])

    if len(tokens) > 0:
        token_table = util.make_table(tokens)
        location_col, location_name = util.find_column(token_table, "location")
        text_col, text_name = util.find_column(token_table, "text")
        char_span = util.make_char_span(location_col, text_col, original_text)

        # Drop location, text columns that is duplicated in char_span
        token_table = token_table.drop([location_name, text_name])

        # Add the span columns to the DataFrames
        token_df = token_table.to_pandas()
        token_df['span'] = char_span
    else:
        char_span = None
        token_df = pd.DataFrame()

    if len(sentence) > 0:
        sentence_table = util.make_table(sentence)
        sentence_df = sentence_table.to_pandas()
        if char_span is not None:
            location_col, _ = util.find_column(sentence_table, "location")
            text_col, _ = util.find_column(sentence_table, "text")
            sentence_char_span = util.make_char_span(location_col, text_col, original_text)
            sentence_span = TokenSpanArray.align_to_tokens(char_span, sentence_char_span)
            sentence_df['span'] = sentence_char_span
            sentence_df['sentence_span'] = sentence_span
    else:
        sentence_df = pd.DataFrame()

    return token_df, sentence_df


def _merge_syntax_dataframes(token_df, sentence_series):

    df = token_df.merge(
        contain_join(
            sentence_series,
            token_df['span'],
            first_name="sentence",
            second_name="span",
        ), how="outer"
    )

    return df


def _make_relations_dataframe(relations, original_text, sentence_span_series):
    if len(relations) == 0:
        return pd.DataFrame()

    table = util.make_table(relations)

    location_cols = {}  # Type: Dict[int, Tuple[Union[Array, ChunkedArray], str]]

    # Separate each argument into a column
    flattened_arguments = []
    drop_cols = []
    for name in table.column_names:
        if name.lower().startswith("arguments"):
            col = pa.concat_arrays(table.column(name).iterchunks())
            assert pa.types.is_list(col.type)

            name_split = name.split('.', maxsplit=1)
            num_arguments = len(col[0])

            value_series = col.values.to_pandas()

            # Separate the arguments into individual columns
            for i in range(num_arguments):
                arg_name = "{}.{}.{}".format(name_split[0], i, name_split[1])
                arg_series = value_series[i::num_arguments]

                arg_array = pa.array(arg_series)

                # If list array is fixed length with 1 element, it can be flattened
                temp = arg_array
                while pa.types.is_list(temp.type):
                    temp = temp.flatten()
                    if len(temp) == len(arg_array):
                        # TODO also need to verify each offset inc by 1?
                        arg_array = temp

                if name.lower().endswith("location"):
                    location_cols[i] = (arg_array, "{}.{}".format(name_split[0], i))

                flattened_arguments.append((arg_array, arg_name))
            drop_cols.append(name)

    # Add the flattened argument columns
    for arg_array, arg_name in flattened_arguments:
        table = table.append_column(arg_name, arg_array)

    # Replace argument location and text columns with spans
    arg_span_cols = {}
    for arg_i, (location_col, arg_prefix) in location_cols.items():
        text_col, text_name = util.find_column(table, "{}.text".format(arg_prefix))
        arg_span_cols["{}.span".format(arg_prefix)] = util.make_char_span(location_col,
                                                                          text_col,
                                                                          original_text)
        drop_cols.extend(["{}.location".format(arg_prefix), text_name])

    add_cols = arg_span_cols.copy()

    # Build the sentence span and drop plain text sentence col
    sentence_col, sentence_name = util.find_column(table, "sentence")
    arg_col_names = list(arg_span_cols.keys())
    if len(arg_col_names) > 0:
        first_arg_span_array = arg_span_cols[arg_col_names[0]]

        sentence_matches = []
        for i, arg_span in enumerate(first_arg_span_array):
            arg_begin = arg_span.begin
            arg_end = arg_span.end
            j = len(sentence_span_series) // 2
            found = False
            while not found:
                sentence_span = sentence_span_series[j]
                if arg_begin >= sentence_span.end:
                    j += 1
                elif arg_end <= sentence_span.begin:
                    j -= 1
                else:
                    contains = [sentence_span.contains(a[i]) for a in arg_span_cols.values()]
                    if not (all(contains) and
                            sentence_span.covered_text == sentence_col[i].as_py()):
                        msg = f"Mismatched sentence span for: {sentence_span}"
                        if not all(contains):
                            msg += f"\nContains Args: {all(contains)}"
                        if sentence_span.covered_text != sentence_col[i].as_py():
                            msg += f"\nSpanText: '{sentence_span.covered_text}'" \
                                   f"\nSentence: '{sentence_col[i]}'"
                        warnings.warn(msg)
                    sentence_matches.append(j)
                    found = True

        relations_sentence = sentence_span_series[sentence_matches]
        add_cols["sentence_span"] = relations_sentence.reset_index(drop=True)
        drop_cols.append(sentence_name)
    else:
        warnings.warn("Could not make sentence span column for Re")

    # Drop columns that have been flattened or replaced by spans
    table = table.drop(drop_cols)

    df = table.to_pandas()

    # Insert additional columns
    for col_name, col in add_cols.items():
        df[col_name] = col

    return df


def _make_relations_dataframe_zero_copy(relations):
    if len(relations) == 0:
        return pd.DataFrame()

    table =util.make_table(relations)

    # Separate each argument into a column
    flattened_arguments = []
    drop_cols = []
    for name in table.column_names:
        if name.lower().startswith("arguments"):
            col = pa.concat_arrays(table.column(name).iterchunks())
            assert pa.types.is_list(col.type)
            is_nested_list = pa.types.is_list(col.type.value_type)

            name_split = name.split('.', maxsplit=1)
            first_list = col[0]
            num_arguments = len(first_list)

            null_count = 0

            # Get the flattened raw values
            raw = col
            offset_arrays = []
            while pa.types.is_list(raw.type):
                offset_arrays.append(raw.offsets)
                null_count += raw.null_count
                raw = raw.flatten()

            # TODO handle lists with null values
            if null_count > 0:
                continue

            # Convert values to numpy
            values = raw.to_numpy(zero_copy_only=False)  # string might copy
            offsets_list = [o.to_numpy() for o in offset_arrays]

            # Compute the length of each list in the array
            value_offsets = offsets_list.pop()
            value_lengths = value_offsets[1:] - value_offsets[:-1]

            # Separate the arguments into individual columns
            for i in range(num_arguments):
                arg_name = "{}.{}.{}".format(name_split[0], i, name_split[1])
                arg_lengths = value_lengths[i::num_arguments]

                # Fixed length arrays can be sliced
                if not is_nested_list or len(np.unique(arg_lengths)) == 1:
                    num_elements = len(first_list[i]) if is_nested_list else 1

                    # Only 1 element so leave in primitive array
                    if not is_nested_list or num_elements == 1:
                        arg_values = values[i::num_arguments]
                        arg_array = pa.array(arg_values)
                    # Multiple elements so put back in a list array
                    else:
                        arg_values = values.reshape([len(col) * num_arguments, num_elements])
                        arg_values = arg_values[i::num_elements]
                        arg_values = arg_values.flatten()
                        arg_offsets = np.cumsum(arg_lengths)
                        arg_offsets = np.insert(arg_offsets, 0, 0)
                        arg_array = pa.ListArray.from_arrays(arg_offsets, arg_values)
                else:
                    # TODO Argument properties with variable length arrays not currently
                    #  supported
                    continue

                flattened_arguments.append((arg_array, arg_name))
            drop_cols.append(name)

    # Add the flattened argument columns
    for arg_array, arg_name in flattened_arguments:
        table = table.append_column(arg_name, arg_array)

    # Drop columns that have been flattened
    table = table.drop(drop_cols)

    return table.to_pandas()


def _make_entity_mentions_dataframe(entities: List,
                                    original_text: str,
                                    apply_standard_schema: bool) -> pd.DataFrame:
    """
    Unroll the records of the "mentions" element of NLU entities into a flat
    DataFrame. Schema of this DataFrame is `_entity_mentions_schema`
    above.
    :param entities: The "entities" section of a parsed NLU response
    :param original_text: Text of the document.  This argument must be provided if there
     are entity mention spans.
    :param apply_standard_schema: Value of the eponymous argument from `parse_response`.
    """
    if 0 == len(entities) or "mentions" not in entities[0].keys():
        # No mentions to unroll. Return an empty DataFrame.
        return util.apply_schema(
            pd.DataFrame(columns=[e[0] for e in _entity_mentions_schema]),
            _entity_mentions_schema, apply_standard_schema)
    if original_text is None:
        raise ValueError(
            "Unable to construct target text for converting entity mentions to spans")
    # Explode out the nested relations containing entity location information.
    # If there was a version of DataFrame.explode() that could handle structs,
    # we would be able to vectorize this operation.
    # Instead we build up the values one row at a time.
    # Some columns come from "parent" entity records, and some columns come from the
    # "child" entity mention records.
    num_parent_cols = len(_entity_mentions_parent_elems)
    parent_cols = [[] for i in range(num_parent_cols)]
    begins = []
    ends = []
    confidences = []
    for e in entities:
        for m in e["mentions"]:
            for i in range(num_parent_cols):
                parent_elem = e[_entity_mentions_parent_names[i]]
                parent_cols[i].append(parent_elem)
            begins.append(m["location"][0])
            ends.append(m["location"][1])
            confidences.append(m["confidence"])  # N.B. confidence of mention, not entity
    # Construct columns, then convert to a DataFrame
    df_cols = {
        _entity_mentions_parent_names[i]: parent_cols[i]
        for i in range(len(_entity_mentions_parent_names))
    }
    df_cols["span"] = SpanArray(original_text, begins, ends)
    df_cols["confidence"] = confidences
    return util.apply_schema(pd.DataFrame(df_cols), _entity_mentions_schema,
                             apply_standard_schema)


def parse_response(response: Dict[str, Any],
                   original_text: str = None,
                   apply_standard_schema: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Parse a Watson NLU response as a decoded JSON string, e.g. dictionary containing requested
    features and convert into a dict of Pandas DataFrames. The following features in the response
    will be converted:
        * entities
        * entity_mentions (elements of the "mentions" field of `response["entities"]`)
        * keywords
        * relations
        * semantic_roles
        * syntax

    For information on getting started with Watson Natural Language Understanding on IBM Cloud, see
    https://cloud.ibm.com/docs/natural-language-understanding?topic=natural-language-understanding-getting-started.
    A Python SDK for authentication and making requests to the service is provided at
    https://github.com/watson-developer-cloud/python-sdk.  Details on the supported features and
    available options when making the request can be found at
    https://cloud.ibm.com/apidocs/natural-language-understanding?code=python#analyze-text.

    .. note:: Additional feature data in response will not be processed

    >>> response = natural_language_understanding.analyze(
    ...     url="https://raw.githubusercontent.com/CODAIT/text-extensions-for-pandas/master/resources/holy_grail.txt",
    ...         return_analyzed_text=True,
    ...         features=Features(
    ...         entities=EntitiesOptions(sentiment=True),
    ...         keywords=KeywordsOptions(sentiment=True, emotion=True),
    ...         relations=RelationsOptions(),
    ...         semantic_roles=SemanticRolesOptions(),
    ...         syntax=SyntaxOptions(sentences=True, tokens=SyntaxOptionsTokens(lemma=True, part_of_speech=True))
    ...     )).get_result()
    >>> dfs = parse_response(response)
    >>> dfs.keys()
    dict_keys(['syntax', 'entities', 'keywords', 'relations', 'semantic_roles'])
    >>> dfs["syntax"].head()
       span part_of_speech      lemma  \
    0  [0, 5): 'Monty'          PROPN    None
    1  [6, 12): 'Python'        PROPN  python

                                                sentence
    0  [0, 273): 'Monty Python and the Holy Grail is ...
    1  [0, 273): 'Monty Python and the Holy Grail is ...

    :param response: A dictionary of features from the IBM Watson NLU response
    :param original_text: Optional original text sent in request, if None will
                          look for "analyzed_text" keyword in response
    :param apply_standard_schema: Return DataFrames with a set schema, whether data
                                  was present in the response or not
    :return: A dictionary mapping feature name to a Pandas DataFrame
    """
    dfs = {}

    if original_text is None and "analyzed_text" in response:
        original_text = response["analyzed_text"]

    # Create the syntax DataFrame
    syntax_response = response.get("syntax", {})
    token_df, sentence_df = _make_syntax_dataframes(syntax_response, original_text)
    sentence_series = sentence_df.get("sentence_span")
    if sentence_series is not None:
        syntax_df = _merge_syntax_dataframes(token_df, sentence_series)
    else:
        syntax_df = pd.concat([token_df, sentence_df], axis=1)
    dfs["syntax"] = util.apply_schema(syntax_df, _syntax_schema, apply_standard_schema)

    if original_text is None and "span" in dfs["syntax"].columns:
        char_span = dfs["syntax"]["span"]
        if isinstance(char_span, SpanArray):
            original_text = dfs["syntax"]["span"].target_text
        else:
            warnings.warn("Did not receive and could not build original text")

    # Create the entities DataFrames
    entities = response.get("entities", [])
    entities_df = util.make_dataframe(entities)
    dfs["entities"] = util.apply_schema(entities_df, _entities_schema,
                                        apply_standard_schema)
    dfs["entity_mentions"] = _make_entity_mentions_dataframe(entities,
                                                             original_text,
                                                             apply_standard_schema)

    # Create the keywords DataFrame
    keywords = response.get("keywords", [])
    keywords_df = util.make_dataframe(keywords)
    dfs["keywords"] = util.apply_schema(keywords_df, _keywords_schema,
                                        apply_standard_schema)

    # Create the relations DataFrame
    relations = response.get("relations", [])
    relations_df = _make_relations_dataframe(relations, original_text, sentence_series)
    dfs["relations"] = util.apply_schema(relations_df, _relations_schema,
                                         apply_standard_schema)

    # Create the semantic roles DataFrame
    semantic_roles = response.get("semantic_roles", [])
    semantic_roles_df = util.make_dataframe(semantic_roles)
    dfs["semantic_roles"] = util.apply_schema(semantic_roles_df, _semantic_roles_schema,
                                              apply_standard_schema)

    if "warnings" in response:
        # TODO: check structure of warnings and improve message
        warnings.warn(response["warnings"])

    return dfs


def make_span_from_entities(char_span: SpanArray,
                            entities_frame: pd.DataFrame,
                            entity_col: str = "text") -> TokenSpanArray:
    """
    Create a token span array for entity text from the entities DataFrame, and an existing
    char span array with tokens from the entire analyzed text.

    :param char_span: Parsed tokens
    :param entities_frame: Entities DataFrame from `parse_response`
    :param entity_col: Column name for the entity text
    :return: TokenSpanArray for matching entities
    """
    entities = entities_frame[entity_col]
    entities_len = entities.str.len()
    begins = []
    ends = []

    i = 0
    while i < len(char_span):
        span = char_span[i]
        text = span.covered_text
        end = i
        num_tokens = 1
        stop = False
        while not stop:
            stop = True
            starts_with = entities.str.startswith(text)
            if any(starts_with):
                # Have a complete match, advance the end index
                if any(entities_len[starts_with] == len(text)):
                    end = i + num_tokens
                # Try the next token
                if i + num_tokens < len(char_span):
                    span = char_span[i + num_tokens]
                    text = text + " " + span.covered_text
                    num_tokens += 1
                    stop = False

        if i != end:
            begins.append(i)
            ends.append(end)
            i += (end - i)
        else:
            i += 1

    return TokenSpanArray(char_span, begins, ends)
