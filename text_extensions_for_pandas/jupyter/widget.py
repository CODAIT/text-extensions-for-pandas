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
from traitlets import HasTraits, Dict, Int, Bool, All, default, observe


def render(dataframe):
    
    # This import ensures proper idomwidget hooks are invoked
    import idom_jupyter

    return DataFrameWidget({
        "dataframe": dataframe
    })

class MetaData(HasTraits):
    selected = Bool().tag(sync=True)
    
    def __repr__(self):
        return f"{self.selected}"
class DataFrameWidget(HasTraits):
    _dataframe = Dict()
    widget = None
    widget_output = ipw.Output()

    def __init__(self, props):
        dataframe = props["dataframe"]
        self._dataframe = dataframe.to_dict("split")
        self._dataframe["columns"].insert(0, "metadata")
        for row_index in range(len(self._dataframe["data"])):
            row_metadata = MetaData(row_index, self.widget_output)
            row_metadata.selected = True
            self._dataframe["data"][row_index].insert(0, row_metadata)
        self.widget = DataFrameWidgetComponent({
            "dataframe": self._dataframe,
            "update_metadata": self.update_metadata
        })
        self.widget.observe(self.print_change, names=All)

    def display(self):
        return ipw.VBox([self.widget_output, self.widget])
    
    def to_dataframe(self):
        return pandas.DataFrame.from_records(self._dataframe["data"], index=self._dataframe["index"], columns=self._dataframe["columns"])

    def update_metadata(self, change):
        index = int(change['owner']._dom_classes[0])
        self._dataframe["data"][index][0].selected = change["new"]

    # Event logging method
    def print_change(self, change):
        with self.widget_output:
            print(change)

def DataFrameWidgetComponent(props):
    """The base component of the dataframe widget"""
    return DataFrameTableComponent(props)

def DataFrameTableComponent(props):
    """Component representing the complete table of a dataframe widget."""
    dataframe = props["dataframe"]
    update_metadata = props["update_metadata"]
    # For each row in the dataframe, create a table with that data

    table_rows = []
    for df_index in range(len(dataframe["data"])):
        df_row = dataframe["data"][df_index]
        table_rows.append(DataFrameTableRowComponent({"df_index": df_index, "df_row": df_row, "columns": dataframe["columns"], "update_metadata": update_metadata}))
    
    return ipw.VBox(table_rows)

def DataFrameTableRowComponent(props):
    """Responsible for returning the HTML representation of the single row defined in the props."""
    row = props["df_row"] # Row of the dataframe
    df_index = props["df_index"]
    columns = props["columns"] # Index of dataframe columns
    update_metadata = props["update_metadata"]
    # Create the metadata controls
    selected_cbox = ipw.Checkbox(value=row[0].selected, indent=False)
    selected_cbox.add_class(str(df_index))
    selected_cbox.observe(update_metadata, names='value')

    # For each column, create a new table header
    table_row_cells = []
    table_row_cells.append(
        ipw.Box(children=[selected_cbox], layout=ipw.Layout(flex='0 0 fit-content', max_width='2em'))
    )
    for column_index in range(1, len(columns)):
        table_row_cells.append(
            ipw.Box(children=[ipw.HTML(f"{str(row[column_index])}")], layout=ipw.Layout(flex='0 1 auto'))
        )
    table_row_cells.append(ipw.Box(layout=ipw.Layout(flex='1 0 auto')))
    return ipw.HBox(table_row_cells)