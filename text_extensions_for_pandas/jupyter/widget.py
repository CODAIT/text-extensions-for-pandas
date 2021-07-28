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
import pandas
import ipywidgets as ipw
from pandas.core.frame import DataFrame
from traitlets import HasTraits, List, Dict


def render(dataframe):
    
    # This import ensures proper idomwidget hooks are invoked
    import idom_jupyter

    return DataFrameWidget({
        "dataframe": dataframe
    })
class DataFrameWidget(HasTraits):
    metadata = List()
    _dataframe = Dict()
    widget = None

    def __init__(self, props):
        dataframe = props["dataframe"]
        self._dataframe = dataframe.to_dict("split")
        self.widget = DataFrameWidgetComponent({
            "dataframe": self._dataframe
        })

    def display(self):
        return self.widget
        
    def _repr_html_(self):
        return "Call this object's display method to view the widget."

def DataFrameWidgetComponent(props):
    """The base component of the dataframe widget"""
    return DataFrameTableComponent(props)

def DataFrameTableComponent(props):
    """Component representing the complete table of a dataframe widget."""
    dataframe = props["dataframe"]
    # For each row in the dataframe, create a table with that data
    table_headers = []
    for column in dataframe["columns"]:
        table_headers.append(f"<th>{column}</th>")

    table_rows = []
    for df_index in range(len(dataframe["data"])):
        df_row = dataframe["data"][df_index]
        table_rows.append(DataFrameTableRowComponent({"df_index": df_index, "df_row": df_row, "columns": dataframe["columns"]}))
    
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
    for column_index in range(len(columns)):
        table_html_pieces.append(f"""
            <td data-column={columns[column_index]}>{row[column_index]}</td>
        """)
    return f"""
    <tr>
        {"".join(table_html_pieces)}
    </tr>
    """