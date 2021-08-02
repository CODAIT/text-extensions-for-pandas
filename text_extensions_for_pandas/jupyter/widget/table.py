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
# table.py
#
# Part of text_extensions_for_pandas
#
# Contains the table elements of the dataframe/spanarray widget
#

import ipywidgets as ipw

def DataFrameTableComponent(widget, dataframe, update_metadata):
    """Component representing the complete table of a dataframe widget."""

    #sheet = ipysheet.sheet(rows=len(dataframe["data"]), columns=len(dataframe["columns"]), row_headers=False, column_headers=False)
    #sheet = ipysheet.sheet(rows=3, columns=4)

    # For each row in the dataframe, create a table with that data
    # table_rows = []
    # for df_index in range(len(dataframe["data"])):
    #     df_row = dataframe["data"][df_index]
    #     # for cell_index in range(len(df_row)):
    #     #     cell = ipysheet.cell(row=df_index, column=cell_index, value=1)
    #     table_rows.append(DataFrameTableRowComponent(row=df_row, index=df_index, columns=dataframe["columns"], update_metadata=update_metadata))
    
    table_columns = []
    for column in dataframe.columns:
        table_columns.append(DataFrameTableColumnComponent(dataframe[column]))

    return ipw.HBox(table_columns)

def DataFrameTableColumnComponent(column):
    column_items = []
    # Column Header
    column_items.append(
        ipw.HTML(f"<b>{column.name}</b>")
    )
    # Column Items
    for item in column:
        column_items.append(
            ipw.HBox(children=[ipw.HTML(f"<div>{str(item)}</div>")], layout=ipw.Layout(justify_content="flex-end", border='1px solid gray', margin='0'))
        )
    return ipw.VBox(children=column_items, layout=ipw.Layout(border='0px solid black'))

###
# QUARANTINE ZONE -----------------------
###
def DataFrameTableRowComponent(row, index, columns, update_metadata):
    """Responsible for returning the HTML representation of the single row defined in the props."""
    # Create the metadata controls
    selected_cbox = ipw.Checkbox(value=bool(row[0].get("selected")), indent=False)
    selected_cbox.add_class(str(index))
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

### --------------------------------------
