#
#  Copyright (c) 2021 IBM Corp.
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

#
# widget.py
#
# Part of text_extensions_for_pandas
#
# Contains the base elements of the dataframe/spanarray widget
#

import idom
import ipywidgets as ipw

def render(dataframe):
    
    # This import ensures proper idomwidget hooks are invoked
    import idom_jupyter

    return DataFrameWidget({
        "dataframe": dataframe
    })

def DataFrameWidget(props):
    """The base component of the dataframe widget"""
    return DataFrameTableComponent(props)

def DataFrameTableComponent(props):
    """Component representing the complete table of a dataframe widget."""
    dataframe = props["dataframe"]
    # For each row in the dataframe, create a table with that data
    table_headers = []
    for column in dataframe.columns:
        table_headers.append(f"<th>{column}</th>")

    table_rows = []
    for df_index, df_row in props["dataframe"].iterrows():
        table_rows.append(DataFrameTableRowComponent({"df_index": df_index, "df_row": df_row, "columns": dataframe.columns}))
    
    table_html = f"""
        <table>
            <tr>
                {"".join(table_headers)}
            </tr>
            {"".join(table_rows)}
        </table>
    """
    return ipw.HTML(table_html)

def DataFrameTableRowComponent(props):
    """Responsible for returning the HTML representation of the single row defined in the props."""
    row = props["df_row"] # Row of the dataframe
    columns = props["columns"] # Index of dataframe columns
    # For each column, create a new table header
    table_html_pieces = []
    for column in columns:
        table_html_pieces.append(f"""
            <td data-column={column}>{row[column]}</td>
        """)
    return f"""
    <tr>
        {"".join(table_html_pieces)}
    </tr>
    """