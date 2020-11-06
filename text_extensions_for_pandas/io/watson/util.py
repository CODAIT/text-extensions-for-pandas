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
# util.py
#
# Internal utility functions for reading ibm_cloud outputs
# not exposed in the public API.


import warnings

import pandas as pd
import pyarrow as pa

from text_extensions_for_pandas import SpanArray


def schema_to_names(schema):
    return [col for col, t in schema]


def apply_schema(df, schema, std_schema_on):
    # TODO: Apply the dtype information in schema, not just the names
    # TODO: Document what this mysterious "std_schema_on" argument does.
    columns = [n for n in schema_to_names(schema) if std_schema_on or n in df.columns]
    return df.reindex(columns=columns)


def find_column(table, column_endswith):
    for name in table.column_names:
        if name.lower().endswith(column_endswith):
            return table.column(name), name
    raise ValueError("Expected {} column but got {}".format(column_endswith, table.column_names))


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


def make_dataframe(records):
    if len(records) == 0:
        return pd.DataFrame()

    table = make_table(records)

    return table.to_pandas()


def build_original_text(text_col, begins):
    # Attempt to build the original text by padding tokens with spaces
    # NOTE: this will not be exactly original text because no newlines or other token separators
    text = ""
    for token, begin in zip(text_col, sorted(begins)):
        if len(text) < begin:
            text += " " * (begin - len(text))
        text += token.as_py()
    return text


def make_char_span(location_col, text_col, original_text):
    """
    Convert a column of begin, end pairs to a SpanArray.

    :param location_col: Arrow array containing (begin, end) tuples of character offset
    :param text_col: Arrow array of strings that should match the target texts of the
      spans in location_col. Used for reconstructing target text when it is not provided.
    :param original_text: Target text for the spans. Optional. If not provided, this
     function will reconstruct target text from the contents of `text_col`.
    """

    # Replace location columns with char and token spans
    if not (pa.types.is_list(location_col.type) and pa.types.is_primitive(location_col.type.value_type)):
        raise ValueError("Expected location column as a list of integers")

    # TODO: assert location is fixed with 2 elements?
    if isinstance(location_col, pa.ChunkedArray):
        location_col = pa.concat_arrays(location_col.iterchunks())

    # Flatten to get primitive array convertible to numpy
    array = location_col.flatten()
    values = array.to_numpy()
    begins = values[0::2]
    ends = values[1::2]

    if original_text is None:
        warnings.warn("Analyzed text was not provided, attempting to reconstruct from tokens, "
                      "however it will not be identical to the original analyzed text.")
        if isinstance(text_col, pa.ChunkedArray):
            text_col = pa.concat_arrays(text_col.iterchunks())
        original_text = build_original_text(text_col, begins)

    return SpanArray(original_text, begins, ends)