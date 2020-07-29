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
# watson_tables.py
#
# I/O functions related to Watson Compare and Comply table processsing on the ibm cloud.
# This service provides analysis of text feature through a request/response API, see
# https://cloud.ibm.com/docs/compare-comply?topic=compare-comply-understanding_tables
# for information on getting started with the service. Details of the provide API and
# available features can be found at https://cloud.ibm.com/apidocs/compare-comply?code=python#extract-a-document-s-tables
# For convience, a Python SDK is available at https://github.com/watson-developer-cloud/python-sdk that
# can be used to authenticate and make requests to the service.

from text_extensions_for_pandas.array import CharSpanArray
import pandas as pd
from typing import *
import pyarrow as pa
import text_extensions_for_pandas.io.watson_util as util

#

_DEFAULT_DROP_COLS = ["column_header_ids", "row_header_ids", "column_index_begin", "column_index_end",
                      "row_index_begin", "row_index_end"]


def _make_headers_df(headers_response):
    headers_df = util.make_dataframe(headers_response)
    headers_df = headers_df[
        ["text", "column_index_begin", "column_index_end", "row_index_begin", "row_index_end", "cell_id",
         "text_normalized"]]
    return headers_df


def _make_body_cells_df(body_cells_response):
    body_cells_df = util.make_dataframe(body_cells_response)
    body_cells_df = body_cells_df[
        ["text", "column_index_begin", "column_index_end", "row_index_begin", "row_index_end", "cell_id",
         "column_header_ids", "column_header_texts", "row_header_ids", "row_header_texts"]]
    return body_cells_df


def _explode_indexes(df, axis_type, drop_original=False):
    # backup version of horizontal explode for if row names or col names are missing
    # uses the range from {axis_type}_index_begin and {axis_type}_index_end as column labels
    df[f'{axis_type}_index'] = [list(range(r[f"{axis_type}_index_begin"], r[f"{axis_type}_index_end"] + 1)) for _, r in
                                df.iterrows()]
    df = df.explode(f'{axis_type}_index')

    if drop_original:
        df.drop(columns=f"{axis_type}_index_begin", inplace=True)
    return df, [f'{axis_type}_index']


def _horiz_explode(df_in, column, drop_original=True):
    # expands list of columns
    df = df_in.copy()
    # expand df.tags into its own dataframe
    tags = df[column].apply(pd.Series)
    tags = tags.rename(columns=lambda x: column + '_' + str(x))
    df = pd.concat([df[:], tags[:]], axis=1)

    if drop_original:
        df.drop(columns=column, inplace=True)
    return df, tags.columns.to_list()


def watson_tables_parse_response(response: Dict[str, Any], table_number=0) -> Dict[str, pd.DataFrame]:
    #
    return_dict = {}

    tables = response.get("tables", [])
    table = tables[table_number]
    plain_text = table.get("text")
    table_title = table.get("text")

    # Create Row headers DataFrame
    row_headers = table.get("row_headers", [])
    if len(row_headers) == 0:
        return_dict["row_headers"] = None
    else:
        row_headers_df = _make_headers_df(row_headers)
        return_dict["row_headers"] = row_headers_df

    # Create Column headers DataFrame
    column_headers = table.get("column_headers", [])
    if len(column_headers) == 0:
        return_dict["column_headers"] = None
    else:
        column_headers_df = _make_headers_df(column_headers)
        return_dict["col_headers"] = column_headers_df

    # Create body cells dataframe
    body_cells = table.get("body_cells", [])
    body_cells_df = _make_body_cells_df(body_cells)
    return_dict["body_cells"] = body_cells_df

    return return_dict


def make_exploded_df(dfs_dict: Dict[str, pd.DataFrame], drop_original: bool = True,
                     explode_row_method: str = None,
                     explode_col_method: str = None
                     ) -> Tuple[pd.DataFrame, list, list]:
    body = dfs_dict["body_cells"]
    if explode_col_method is None:
        if (not dfs_dict["col_headers"] is None) and len(dfs_dict["col_headers"]) != 0:
            explode_col_method = "title"
        else:
            explode_col_method = "index"

    if explode_row_method is None:
        if (not dfs_dict["row_headers"] is None) and len(dfs_dict["row_headers"]) != 0:
            explode_row_method = "title"
        else:
            explode_row_method = "index"

    if explode_col_method == "title":
        exploded, col_header_names = _horiz_explode(body, "column_header_texts", drop_original=drop_original)
    elif explode_col_method == "title_id":
        exploded, col_header_names = _horiz_explode(body, "column_header_ids", drop_original=drop_original)
    elif explode_col_method == "index":
        exploded, col_header_names = _explode_indexes(body, "column", drop_original=drop_original)
    else :
        exploded = body
        col_header_names = []

    if explode_row_method == "title":
        exploded, row_header_names = _horiz_explode(exploded, "row_header_texts", drop_original=drop_original)
    elif explode_row_method == "title_id":
        exploded, row_header_names = _horiz_explode(exploded, "row_header_ids", drop_original=drop_original)
    elif explode_row_method == "index":
        exploded, row_header_names = _explode_indexes(exploded, "row", drop_original=drop_original)
    else:
        exploded = exploded
        row_header_names = []

    return exploded, row_header_names, col_header_names


def make_table_from_exploded_df(exploded_df: pd.DataFrame, row_heading_cols, column_heading_cols,
                                value_col: str = "text", concat_with: str = " | ") -> object:
    table = exploded_df.pivot_table(index=row_heading_cols, columns=column_heading_cols, values=value_col,
                                   aggfunc=(lambda a: concat_with.join(a)))
    row_nones = [ None for _ in range(len(row_heading_cols))]
    col_nones = [None for _ in range(len(column_heading_cols))]

    return table.rename_axis(index=row_nones, columns=col_nones)


def make_table(dfs_dict: Dict[str, pd.DataFrame], value_col="text", row_explode_by: str = None,
               col_explode_by: str = None, concat_with: str = " | "):
    exploded, row_heading_names, col_heading_names = make_exploded_df(dfs_dict, explode_row_method= row_explode_by,
                                                                      explode_col_method= col_explode_by)
    return make_table_from_exploded_df(exploded, row_heading_names, col_heading_names,
                                       value_col=value_col, concat_with=concat_with)
