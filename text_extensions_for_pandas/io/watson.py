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
# watson.py
#
# I/O functions related to Watson Natural Language Understanding on the IBM Cloud.
# TODO: add links here and docstrings

import numpy as np
import pandas as pd
import pyarrow as pa

from text_extensions_for_pandas.array import CharSpanArray, TokenSpanArray
from text_extensions_for_pandas.spanner import contain_join


def _flatten_struct(struct_array, parent_name=None):
    arrays = struct_array.flatten()
    fields = [f for f in struct_array.type]
    for array, field in zip(arrays, fields):
        name = field.name if parent_name is None else parent_name + "." + field.name
        if pa.types.is_struct(array.type):
            for child_array, child_name in _flatten_struct(array, name):
                yield child_array, child_name
        elif pa.types.is_list(array.type) and pa.types.is_struct(array.type.value_type):
            struct = array.flatten()
            for child_array, child_name in _flatten_struct(struct, name):
                list_array = pa.ListArray.from_arrays(array.offsets, child_array)
                yield list_array, child_name
        else:
            yield array, name


def _make_table(records):
    arr = pa.array(records)
    assert pa.types.is_struct(arr.type)
    arrays, names = zip(*_flatten_struct(arr))
    return pa.Table.from_arrays(arrays, names)


def _find_column(table, column_endswith):
    for name in table.column_names:
        if name.lower().endswith(column_endswith):
            return table.column(name), name
    raise ValueError("Expected {} column but got {}".format(column_endswith, table.column_names))


def _make_dataframe(records):
    if len(records) == 0:
        # TODO: fill in with expected schema
        return pd.DataFrame()

    table = _make_table(records)
    return table.to_pandas()


def _build_original_text(text_col, begins):
    # Build the covered text, TODO: ok to assume begin is sorted?
    text = ""
    for token, begin in zip(text_col, begins):
        if len(text) < begin:
            text += " " * (begin - len(text))
        text += token.as_py()
    return text


def _make_char_span(location_col, text_col, original_text):

    # Replace location columns with char and token spans
    if not (pa.types.is_list(location_col.type) and pa.types.is_primitive(location_col.type.value_type)):
        raise ValueError("Expected location column as a list of integers")

    # TODO: assert location is fixed with 2 elements?
    if isinstance(location_col, pa.ChunkedArray):
        location_col = pa.concat_arrays(location_col.iterchunks())
    if isinstance(text_col, pa.ChunkedArray):
        text_col = pa.concat_arrays(text_col.iterchunks())

    # Flatten to get primitive array convertible to numpy
    array = location_col.flatten()
    values = array.to_numpy()
    begins = values[0::2]
    ends = values[1::2]

    if original_text is None:
        original_text = _build_original_text(text_col, begins)

    return CharSpanArray(original_text, begins, ends)


def _make_syntax_dataframes(syntax_response, original_text):
    tokens = syntax_response.get("tokens", [])
    sentence = syntax_response.get("sentences", [])

    if len(sentence) == 0 and len(tokens) == 0:
        # TODO: fill in with expected schema
        return pd.DataFrame()

    token_table = _make_table(tokens)
    location_col, location_name = _find_column(token_table, "location")
    text_col, text_name = _find_column(token_table, "text")
    char_span = _make_char_span(location_col, text_col, original_text)
    token_span = TokenSpanArray.from_char_offsets(char_span)

    # Drop location, text columns that is duplicated in char_span
    token_table = token_table.drop([location_name, text_name])

    sentence_table = _make_table(sentence)
    location_col, _ = _find_column(sentence_table, "location")
    text_col, _ = _find_column(sentence_table, "text")
    sentence_char_span = _make_char_span(location_col, text_col, original_text)
    sentence_span = TokenSpanArray.align_to_tokens(char_span, sentence_char_span)

    # Add the span columns to the DataFrames
    token_df = token_table.to_pandas()
    token_df['char_span'] = char_span
    token_df['token_span'] = token_span

    sentence_df = sentence_table.to_pandas()
    sentence_df['char_span'] = sentence_char_span
    sentence_df['sentence_span'] = sentence_span

    return token_df, sentence_df


def _merge_syntax_dataframes(token_df, sentence_series):

    df = token_df.merge(
        contain_join(
            sentence_series,
            token_df['token_span'],
            first_name="sentence",
            second_name="token_span",
        ), how="outer"
    )

    return df


def _make_relations_dataframe(relations, original_text, sentence_span_series):
    if len(relations) == 0:
        # TODO: fill in with expected schema
        return pd.DataFrame()

    table = _make_table(relations)

    location_cols = {}

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
        text_col, text_name = _find_column(table, "{}.text".format(arg_prefix))
        arg_span_cols["{}.span".format(arg_prefix)] = _make_char_span(location_col, text_col, original_text)
        drop_cols.extend(["{}.location".format(arg_prefix), text_name])

    add_cols = arg_span_cols.copy()

    # Build the sentence span and drop plain text sentence col
    sentence_col, sentence_name = _find_column(table, "sentence")
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
                    if not (all(contains) and sentence_span.covered_text == sentence_col[i]):
                        raise ValueError("Mismatched sentence span")  # TODO issue warning
                    sentence_matches.append(j)
                    found = True

        relations_sentence = sentence_span_series[sentence_matches]
        add_cols["sentence_span"] = relations_sentence.reset_index(drop=True)
        drop_cols.append(sentence_name)
    else:
        pass  # TODO can't make sentence span, show warning?

    # Drop columns that have been flattened or replaced by spans
    table = table.drop(drop_cols)

    df = table.to_pandas()

    # Insert additional columns
    for col_name, col in add_cols.items():
        df[col_name] = col

    return df


def _make_relations_dataframe_zero_copy(relations):
    if len(relations) == 0:
        # TODO: fill in with expected schema
        return pd.DataFrame()

    table = _make_table(relations)

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
                    # TODO Argument properties with variable length arrays not currently supported
                    continue

                flattened_arguments.append((arg_array, arg_name))
            drop_cols.append(name)

    # Add the flattened argument columns
    for arg_array, arg_name in flattened_arguments:
        table = table.append_column(arg_name, arg_array)

    # Drop columns that have been flattened
    table = table.drop(drop_cols)

    return table.to_pandas()


def watson_nlu_parse_response(response, original_text=None):

    dfs = {}

    if original_text is None and "analyzed_text" in response:
        original_text = response["analyzed_text"]

    # Create the syntax DataFrame
    syntax_response = response.get("syntax", {})
    token_df, sentence_df = _make_syntax_dataframes(syntax_response, original_text)
    sentence_series = sentence_df["sentence_span"]
    dfs["syntax"] = _merge_syntax_dataframes(token_df, sentence_series)

    if original_text is None and "char_span" in dfs["syntax"].columns:
        original_text = dfs["syntax"]["char_span"].target_text

    # Create the entities DataFrame
    entities = response.get("entities", [])
    dfs["entities"] = _make_dataframe(entities)

    # Create the keywords DataFrame
    keywords = response.get("keywords", [])
    dfs["keywords"] = _make_dataframe(keywords)

    # Create the relations DataFrame
    relations = response.get("relations", [])
    dfs["relations"] = _make_relations_dataframe(relations, original_text, sentence_series)

    # Create the semantic roles DataFrame
    semantic_roles = response.get("semantic_roles", [])
    dfs["semantic_roles"] = _make_dataframe(semantic_roles)

    return dfs


def make_span_from_entities(entities_frame, entity_col, char_span):
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
