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
    _dataframe = Dict()
    widget = None

    def __init__(self, props):
        dataframe = props["dataframe"]
        self._dataframe = dataframe.to_dict("split")
        self._dataframe["columns"].insert(0, "metadata")
        for row_index in range(len(self._dataframe["data"])):
            self._dataframe["data"][row_index].insert(0, 0)
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

    table_rows = []
    for df_index in range(len(dataframe["data"])):
        df_row = dataframe["data"][df_index]
        table_rows.append(DataFrameTableRowComponent({"df_index": df_index, "df_row": df_row, "columns": dataframe["columns"]}))
    
    return ipw.VBox(table_rows)

def DataFrameTableRowComponent(props):
    """Responsible for returning the HTML representation of the single row defined in the props."""
    row = props["df_row"] # Row of the dataframe
    columns = props["columns"] # Index of dataframe columns
    # For each column, create a new table header
    table_row_cells = []
    table_row_cells.append(
        ipw.Box(children=[ipw.Checkbox(value=row[0])], layout=ipw.Layout(flex='0 0 auto'))
    )
    for column_index in range(1, len(columns)):
        table_row_cells.append(
            ipw.HTML(f"{str(row[column_index])}")
        )
    return ipw.HBox(table_row_cells)