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
import regex
import text_extensions_for_pandas.io.watson_util as util

#


def _make_headers_df(headers_response):
    """
    parses the headers portion of the watson response and creates the header dataframe.
    :param headers_response: the ``row_header`` or ``column_header`` array as returned from the watson response
    :return: the completed header dataframe
    """

    headers_df = util.make_dataframe(headers_response)
    headers_df = headers_df[
        ["text", "column_index_begin", "column_index_end", "row_index_begin", "row_index_end", "cell_id",
         "text_normalized"]]
    return headers_df


def _make_body_cells_df(body_cells_response):
    """
    parses the body_cells portion of the watson response and creates the body_cells dataframe.
    :param body_cells_response: the "body cells" array as returned from the watson response
    :return: the completed body_cells dataframe
    """
    body_cells_df = util.make_dataframe(body_cells_response)
    if not "attributes.type" in body_cells_df.columns.to_list():
        body_cells_df["attributes.type"] = None
        body_cells_df["attributes.text"] = None
    body_cells_df = body_cells_df[
        ["text", "column_index_begin", "column_index_end", "row_index_begin", "row_index_end", "cell_id",
         "column_header_ids", "column_header_texts", "row_header_ids", "row_header_texts",
         "attributes.text", "attributes.type"]]
    return body_cells_df


def _explode_indexes(df, axis_type, drop_original=False):
    """
    Performs an operation similar to explode, but uses the index of the cell as the differentiator.
    :param df: the input dataframe of body cells
    :param axis_type: either ``"row"`` or ``"column"``, determining which axis to be operating on.
    :param drop_original: Whether or not to drop the original column. Defaults to false
    :return: the original dataframe, modified as expected, and the name of the column created.
    """
    df[f'{axis_type}_index'] = [list(range(r[f"{axis_type}_index_begin"], r[f"{axis_type}_index_end"] + 1)) for _, r in
                                df.iterrows()]
    df = df.explode(f'{axis_type}_index')

    if drop_original:
        df.drop(columns=f"{axis_type}_index_begin", inplace=True)
    return df, [f'{axis_type}_index']


def _horiz_explode(df_in, column, drop_original=True):
    """
    Horizontal explode is the default way of 'exploding' the dataframe. It taes a dataframe, and a column that needs
    to be exploded. All list type elements in that column will be seperated into new columns of name ``{
    original_column_name}_{n}``, where n is the position of the elment in the list.

     So the first element in a list in
    ``"row_header"`` column would end up in the ``"row_header_0"`` column, and the second would be in the
    ``"row_header_1`` column etc.

    :param df_in: the dataframe to explode
    :param column: the name of the column as a string to explode. The column should contain listlike elements
    :param drop_original: bool. if ``True``, the original column is removed, otherwise it is left as is.
    :return: the exploded dataframe, and the list of the names of the columns created, as a tuple
    """
    # expands list of columns
    df = df_in.copy()
    # expand df.tags into its own dataframe
    tags = df[column].apply(pd.Series)
    tags = tags.rename(columns=lambda x: column + '_' + str(x))
    df = pd.concat([df[:], tags[:]], axis=1)

    if drop_original:
        df.drop(columns=column, inplace=True)
    return df, tags.columns.to_list()


def _explode_by_concat(df_in, column, agg_by: str = " "):
    def agg_horiz_headers(series):
        series[column] = agg_by.join(series[column])
        return series

    df = df_in.copy()
    df = df.apply(agg_horiz_headers, axis=1)
    return df, [column]


def _generate_is_numeric_table(exploded_df, rows, cols):
    list_of_numeric_labels = ["Number", "Percentage"]
    exploded_df['attributes.type'] = exploded_df['attributes.type'].apply(
        lambda a: a[0] if pd.api.types.is_list_like(a) and len(a) != 0 else a if type(a) == str else "None")

    numerics = make_table_from_exploded_df(exploded_df, rows, cols, value_col="attributes.type")
    numeric_cells = numerics == list_of_numeric_labels[0]
    for label in list_of_numeric_labels[1:]:
        numeric_cells = numeric_cells | (numerics == label)
    return numeric_cells


def _infer_numeric_rows_cols(exploded_df, row_names, col_names):
    """
    intakes the exploded format of the dataframe, and uses it to infer which set of rows or columns are supposed to be
        numeric, then returns them as a list
    :param exploded_df: the exploded dataframe, (penultimate step of table reconstruction)
    :param row_names: the header names of columns in the exploded dataframe indicating row headers,
            (as returned from ``make_exploded_df()``)
    :param col_names: the header names of columns in the exploded dataframe indicating column headers,
            (as returned from ``make_exploded_df()``)
    :return: a tuple, in the form ``(numeric_rows, numeric_columns)`` where one of ``numeric_rows`` or
            ``numeric_columns`` is an empty list, and the other contains all row/columns that are of numeric type
    """
    list_of_numeric_labels = ["Number", "Percentage"]

    numeric_cells = _generate_is_numeric_table(exploded_df, row_names, col_names)

    cols = []
    rows = []
    axis = None

    # if rows are unnamed they very likely are not numeric
    if row_names == ['row_index'] and not col_names == ['column_index']:
        axis = 0  # row centric
    elif col_names == ['column_index'] and not row_names == ['row_index']:
        axis = 1  # col centric
    else:
        # columns/rows with higher variance between numeric and non-numeric less likely to be the primary direction
        col_score = numeric_cells.std(axis=0).mean()
        row_score = numeric_cells.std(axis=1).mean()
        axis = 1 if col_score > row_score else 0

    is_num_scores = numeric_cells.sum(axis=axis) / numeric_cells.shape[axis]
    if axis == 1:
        rows = (is_num_scores[is_num_scores > .9].index.to_list())
    elif axis == 0:
        cols = (is_num_scores[is_num_scores > .9].index.to_list())

    return rows, cols


def _convert_val_to_numeric(val, cast_type=float, regex_exp='[^0-9.]'):
    """
    converts a single value to a numeric type as specified, and removes all non-numeric characters
        If this is not possible, a warning is printed to the commandline and pd.NA is returned instead
    :param val: The value to be converted (a string)
    :param cast_type: the type to convert the value to
    :param regex_exp: the regex expression to remove non-numeric characters. this can be changed to change the decimal
                    point or to allow certian special characters
    :return: an int or float, extracted from the input val
    """

    try:
        ans = cast_type(regex.sub(regex_exp, '', val))
    except ValueError:
        ans = pd.NA
        print(f"ERROR READING VALUE:\"{val}\"\t Filling with <NA>")
    return ans


def _convert_labelled_numeric_items(table, exploded_df, row_header_cols, col_header_cols, decimal_pt='.',
                                    cast_type=float):
    """
    converts all elements in the table that are tagged as numeric type to a numeric type
    :param table:           The table to convert, as a pandas DataFrame
    :param exploded_df:     The exploded DataFrame, from last step of reconstruction
    :param row_header_cols: The names of the headers in the exploded dataframe that refer to row headers
    :param col_header_cols: The names of the headers in the exploded dataframe that refer to column headers
    :param decimal_pt:      the type of the decimal point being used
    :param cast_type:       The type that the element is to be cast to. Defaults to ``float``
    :return:        the converted DataFrame
    """
    def convert_val(val):
        return _convert_val_to_numeric(val, cast_type, f'[^0-9{decimal_pt}]')

    numeric_cells = _generate_is_numeric_table(exploded_df, rows, cols)

    for i in range(table.shape[0]):
        table.iloc[i, :][numeric_cells.iloc[i, :]] = table.iloc[i, :][numeric_cells.iloc[i, :]].apply(convert_val)
    return table


def _convert_cols_to_numeric(df_in: pd.DataFrame, columns=None, rows=None, decimal_pt='.',
                             cast_type=float) -> pd.DataFrame:
    """
    converts inputted columns or rows to numeric format.
        if none are given, it converts all elements to numeric types
        converts to type specified, if not, defaults to ``float`` type
    :param df_in:       dataframe, (table type) to convert
    :param columns:     columns to convert to numeric type, as a list of strings
    :param rows:        rows to convert to numeric type, as a list of strings
    :param decimal_pt:  what symbol is being used as the decimal point (typically ``"."`` or ``","``
    :param cast_type:   type to cast the object to, as a class. Defaults to ``float``
    :return:            the converted table.
    """
    # converts inputted columns or rows to numeric format.
    # if none are given, it converts all elements to numeric types
    # converts to type given
    if columns is None and rows is None:
        columns = df_in.columns
    if columns is None:
        columns = []
    if rows is None:
        rows = []
    df = df_in.copy()

    def convert_val(val):
        return _convert_val_to_numeric(val, cast_type, f'[^0-9{decimal_pt}]')

    for column in columns:
        df[column] = df[column].apply(convert_val)
    for row in rows:
        df[row] = df[row].apply(convert_val)

    return df


def watson_tables_parse_response(response: Dict[str, Any], select_table=None) -> Dict[str, pd.DataFrame]:
    """
    Parse a response from Watson Tables Understanding as a decoded JSON string. e.g.
     dictionary containing requested features and convert into a dict of Pandas DataFrames.
     The following features will be converted from the response:
        * Row headers
        * Column headers
        * Body cells

    More information on using Watson Table Extraction or the Compare and Comply API, see
    https://cloud.ibm.com/docs/compare-comply?topic=compare-comply-understanding_tables
    More infomration about available features can be found at
    https://cloud.ibm.com/apidocs/compare-comply?code=python#extract-a-document-s-tables


    :param response: A dictionary of features returned by the IBM Watson Compare and Comply API
    :param table_number: Defaults to analyzing the first table, input a number here to analyze the nth table
    :return: A dictionary mapping feature names ("row_headers", "col_headers", "body_cells")
                    to Pandas DataFrames
    """
    return_dict = {}
    tables = response.get("tables", [])

    table_number = None
    if select_table is int:
        table_number = select_table
    elif select_table is str:
        for i, table in enumerate(tables):
            if table.get("title", str).lower() == select_table.lower():
                table_number = i

    if table_number is None:
        table_number = 0

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
                     explode_col_method: str = None, keep_all_cols: bool = False,
                     ) -> Tuple[pd.DataFrame, list, list]:
    """
    Creates a value-attribute mapping, mapping the column values to header or row number values
    this is a preliminary stage to creating the final table, but may be a useful intermediary

    :param dfs_dict: The dictionary of {features : DataFrames} returned by watson_tables_parse_response
    :param drop_original: drop the original column location information. defaults to True
    :param explode_row_method: If specified, set the method used to explode rows, instead of the default logic being
                                applied:
                                if "title", the title field will be used to arrange rows
                                if "title_id", the title_id feild will be used to arrange rows
                                if "index", the row / column locations given will be used to arrange rows
    :param explode_col_method: if specified, set the method used to explode columns, instead of the default logic bing
                                applied
                                if "title", the title field will be used to arrange rows
                                if "title_id", the title_id feild will be used to arrange rows
                                if "index", the row / column locations given will be used to arrange rows
    :param keep_all_cols: if false, keep only attributes necessary for constructing final table.
                                gets overridden if drop_original is False.
    :return: a table mapping values to attributes (either headings or row numbers if no headings exist)
    """

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
    elif explode_col_method == "concat":
        exploded, col_header_names = _explode_by_concat(body, "column_header_texts")
    else:
        exploded = body
        col_header_names = []

    if explode_row_method == "title":
        exploded, row_header_names = _horiz_explode(exploded, "row_header_texts", drop_original=drop_original)
    elif explode_row_method == "title_id":
        exploded, row_header_names = _horiz_explode(exploded, "row_header_ids", drop_original=drop_original)
    elif explode_row_method == "index":
        exploded, row_header_names = _explode_indexes(exploded, "row", drop_original=drop_original)
    elif explode_row_method == "concat":
        exploded, row_header_names = _explode_by_concat(exploded, "row_header_texts")
    else:
        exploded = exploded
        row_header_names = []

    if drop_original and not keep_all_cols:
        cols_to_keep = ["text", "attributes.type"] + row_header_names + col_header_names
        exploded = exploded[cols_to_keep]

    return exploded, row_header_names, col_header_names


def make_table_from_exploded_df(exploded_df: pd.DataFrame, row_heading_cols, column_heading_cols, dfs_dict=None,
                                value_col: str = "text", concat_with: str = " | ",
                                convert_numeric_items=False) -> pd.DataFrame:
    """
    takes in the exploded dataframe, and converts it into the reconstructed table

    :param exploded_df: The exploded dataframe, as returned by `make_exploded_df`
    :param row_heading_cols: the names of the columns referring to row headings (as outputted from make_exploded_df())
    :param column_heading_cols: the names of the columns referring to column headings
                                (as outputted from make_exploded_df())
    :param value_col: the name of the column to use for the value of each cell. Defaults to 'text'
    :param concat_with: the delimiter to use when concatinating duplicate entries.
                            Using an empty string, "" will fuse entries
    :param dfs_dict: Dictionary parsed from initial step of table reconstruction. Is required to re-order columns into
                     their original format. If not, the reordering will not take place
    :param convert_numeric_items: if True, rows or columns with numeric items will be detected and converted
                                  to floats or ints
    :return: the reconstructed table. should be a 1:1 translation of original table, but both machine and human readable
    """
    table = exploded_df.pivot_table(index=row_heading_cols, columns=column_heading_cols, values=value_col,
                                    aggfunc=(lambda a: concat_with.join(a)))
    row_nones = [None for _ in range(len(row_heading_cols))]
    col_nones = [None for _ in range(len(column_heading_cols))]

    if convert_numeric_items:
        num_rows, num_cols = _infer_numeric_rows_cols(exploded_df, row_heading_cols, column_heading_cols)
        table = _convert_cols_to_numeric(table, num_cols, num_rows)

    if column_heading_cols != ["column_index"] and len(column_heading_cols) == 1 and dfs_dict is not None:
        col_headings = table.columns.to_list()
        cols = dfs_dict["col_headers"][dfs_dict["col_headers"]["text"].isin(col_headings)]
        cols = cols.sort_values("column_index_begin")
        col_headings_sorted = cols["text"].to_list()
        table = table[col_headings_sorted]

    if row_heading_cols != ["row_index"] and len(column_heading_cols) == 1 and dfs_dict is not None:
        row_headings = table.index.to_list()
        rows = dfs_dict["row_headers"][dfs_dict["row_headers"]["text"].isin(row_headings)]
        rows = rows.sort_values("row_index_begin")
        row_headings_sorted = rows["text"].to_list()
        table = table.reindex(row_headings_sorted)

    return table.rename_axis(index=row_nones, columns=col_nones)


def make_table(dfs_dict: Dict[str, pd.DataFrame], value_col="text", row_explode_by: str = None,
               col_explode_by: str = None, concat_with: str = " | ", convert_numeric_items=True):
    """
    Runs the end-to-end process of creating the table, starting with the parsed response from the Compare & Comply or
    Watson Discovery engine, and returns the completed table.

    :param dfs_dict: The dictionary of {features : DataFrames} returned by watson_tables_parse_response
    :param value_col: Which column to use as values. by default "text"
    :param row_explode_by: If specified, set the method used to explode rows, instead of the default logic being applied
                                if "title", the title field will be used to arrange rows
                                if "title_id", the title_id feild will be used to arrange rows
                                if "index", the row / column locations given will be used to arrange rows
    :param col_explode_by: if specified, set the method used to explode columns, instead of the default logic bing applied
                                if "title", the title field will be used to arrange rows
                                if "title_id", the title_id feild will be used to arrange rows
                                if "index", the row / column locations given will be used to arrange rows
    :param concat_with: the delimiter to use when concatinating duplicate entries. Using an empty string, "" will fuse entries
    :param convert_numeric_items: if `True`, auto-detect and convert numeric rows and columns to numeric datatypes
    :return: the reconstructed table. should be a 1:1 translation of original table

    """

    exploded, row_heading_names, col_heading_names = make_exploded_df(dfs_dict, explode_row_method=row_explode_by,
                                                                      explode_col_method=col_explode_by,
                                                                      keep_all_cols=True)
    return make_table_from_exploded_df(exploded, row_heading_names, col_heading_names,
                                       value_col=value_col, concat_with=concat_with,
                                       convert_numeric_items=convert_numeric_items, dfs_dict=dfs_dict)
