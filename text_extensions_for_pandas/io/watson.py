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

import pandas as pd
import pyarrow as pa

from text_extensions_for_pandas.array import CharSpanArray, TokenSpanArray
from text_extensions_for_pandas.spanner import contain_join


def watson_nlu_parse_response(response):

    def flatten_struct(struct_array, parent_name=None):
        arrays = struct_array.flatten()
        fields = [f for f in struct_array.type]
        for array, field in zip(arrays, fields):
            name = field.name if parent_name is None else parent_name + "." + field.name
            if pa.types.is_struct(array.type):
                for child_array, child_name in flatten_struct(array, name):
                    yield child_array, child_name
            elif pa.types.is_list(array.type) and pa.types.is_struct(array.type.value_type):
                struct = array.flatten()
                for child_array, child_name in flatten_struct(struct, name):
                    list_array = pa.ListArray.from_arrays(array.offsets, child_array)
                    yield list_array, child_name
            else:
                yield array, name

    def make_table(records):
        arr = pa.array(records)
        assert pa.types.is_struct(arr.type)
        arrays, names = zip(*flatten_struct(arr))
        return pa.Table.from_arrays(arrays, names)

    def find_column(table, column):
        for name in table.column_names:
            if name.lower().endswith(column):
                return table.column(name)
        return None

    def make_dataframe(records):
        if len(records) == 0:
            # TODO: fill in with expected schema
            return pd.DataFrame()

        table = make_table(records)
        df = table.to_pandas()
        return df

    def make_char_span(table):
        location_col = find_column(table, "location")
        text_col = find_column(table, "text")

        # Replace location columns with char and token spans
        if location_col is None or text_col is None or \
                not pa.types.is_list(location_col.type) or not pa.types.is_primitive(location_col.type.value_type):
            raise ValueError("Expected location column as a list of integers")

        # TODO: assert location is fixed with 2 elements?
        location_col = pa.concat_arrays(location_col.iterchunks())
        text_col = pa.concat_arrays(text_col.iterchunks())

        # Flatten to get primitive array convertible to numpy
        array = location_col.flatten()
        values = array.to_numpy()
        begins = values[0::2]
        ends = values[1::2]

        # Build the covered text, TODO: ok to assume begin is sorted?
        text = ""
        for token, begin in zip(text_col, begins):
            if len(text) < begin:
                text += " " * (begin - len(text))
            text += token.as_py()

        char_span = CharSpanArray(text, begins, ends)

        return char_span

    def make_syntax_dataframes(syntax_response):
        tokens = syntax_response.get("tokens", [])
        sentence = syntax_response.get("sentences", [])

        if len(sentence) == 0 and len(tokens) == 0:
            # TODO: fill in with expected schema
            return pd.DataFrame()

        token_table = make_table(tokens)
        char_span = make_char_span(token_table)
        token_span = TokenSpanArray.from_char_offsets(char_span)

        sentence_table = make_table(sentence)
        sentence_char_span = make_char_span(sentence_table)
        sentence_span = TokenSpanArray.align_to_tokens(char_span, sentence_char_span)

        # TODO: drop location, text columns

        # Add the span columns to the DataFrames
        token_df = token_table.to_pandas()
        token_df['char_span'] = char_span
        token_df['token_span'] = token_span

        sentence_df = sentence_table.to_pandas()
        sentence_df['char_span'] = sentence_char_span

        df = token_df.merge(
            contain_join(
                pd.Series(sentence_span),
                token_df['token_span'],
                first_name="sentence",
                second_name="token_span",
            ), how="outer"
        )

        return df

    dfs = {}

    # Create the entities DataFrame
    entities = response.get("entities", [])
    dfs["entities"] = make_dataframe(entities)

    # Create the keywords DataFrame
    keywords = response.get("keywords", [])
    dfs["keywords"] = make_dataframe(keywords)

    # Create the relations DataFrame
    relations = response.get("relations", [])
    dfs["relations"] = make_dataframe(relations)

    # Create the semantic roles DataFrame
    semantic_roles = response.get("semantic_roles", [])
    dfs["semantic_roles"] = make_dataframe(semantic_roles)

    # Create the syntax DataFrame
    syntax_response = response.get("syntax", {})
    dfs["syntax"] = make_syntax_dataframes(syntax_response)

    return dfs
