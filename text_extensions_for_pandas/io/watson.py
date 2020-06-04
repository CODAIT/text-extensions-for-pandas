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
        if len(records) > 0:
            table = make_table(records)
            df = table.to_pandas()
            return df
        else:
            # TODO: fill in with expected schema
            return pd.DataFrame()

    def make_dataframe_with_spans(records):
        if len(records) > 0:
            table = make_table(records)

            char_span = None
            location_col = find_column(table, "location")
            text_col = find_column(table, "text")

            # Replace location columns with char and token spans
            if location_col is not None and text_col is not None and \
                pa.types.is_list(location_col.type) and pa.types.is_primitive(location_col.type.value_type):

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

                # TODO: drop location column?

            df = table.to_pandas()

            # Add the span columns to the DataFrame
            if char_span is not None:
                # TODO token_span = TokenSpanArray.al(char_span)
                df['char_span'] = char_span

            return df
        else:
            # TODO: fill in with expected schema
            return pd.DataFrame()

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
    syntax = response.get("syntax", {})
    sentence = syntax.get("sentences", [])
    dfs["syntax.sentence"] = make_dataframe_with_spans(sentence)
    tokens = syntax.get("tokens", [])
    dfs["syntax.tokens"] = make_dataframe_with_spans(tokens)

    return dfs
