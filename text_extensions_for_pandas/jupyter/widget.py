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
import bisect
import collections


def render(dataframe, **kwargs):
    
    # This import ensures proper idomwidget hooks are invoked

    return DataFrameWidget(dataframe=dataframe, **kwargs)

class DataFrameWidget(HasTraits):

    _dataframe_dict = Dict()
    _dtypes = None
    widget = None
    widget_output = ipw.Output()

    def __init__(self, dataframe, metadata_column=None):

        self._df = dataframe.copy(deep=True)
        self._dataframe_dict = dataframe.to_dict("split")
        self._dtypes = dataframe.dtypes

        # Initialize Metadata
        if metadata_column:
            self._metadata_column = metadata_column
        else:
            self._dataframe_dict["columns"].insert(0, "metadata")

            for row_index in range(len(self._dataframe_dict["data"])):
                row_metadata = {
                    "selected": Bool(False).tag(sync=True)
                }
                # When we switch to pure dataframe, insert into dataframe instead of dict
                self._dataframe_dict["data"][row_index].insert(0, row_metadata)

        # Initialize Widget        
        self.widget = DataFrameWidgetComponent(widget=self, dataframe=self._dataframe_dict, dtypes=self._dtypes, update_metadata=self.update_metadata)
        self.widget.observe(self.print_change, names=All)

    def display(self):
        return ipw.VBox([self.widget_output, self.widget])
    
    def to_dataframe(self):
        return pandas.DataFrame.from_records(self._dataframe_dict["data"], index=self._dataframe_dict["index"], columns=self._dataframe_dict["columns"])

    def update_metadata(self, change):
        index = int(change['owner']._dom_classes[0])
        self._dataframe_dict["data"][index][0]["selected"] = change["new"]

    # Event logging method
    def print_change(self, change):
        with self.widget_output:
            print(change)

def DataFrameWidgetComponent(widget, dataframe, dtypes, update_metadata):
    """The base component of the dataframe widget"""

    with widget.widget_output:
        print("Printing widget")

    span_column = None
    # Check if any of the columns are of dtype SpanArray or TokenSpanArray
    for index in range(1,len(dataframe["columns"])):
        column = dataframe["columns"][index]
        with widget.widget_output:
            print(f"{column}: {str(dtypes[column])}")
        
        if span_column == None and (str(dtypes[column]) == "SpanDtype" or str(dtypes[column]) == "TokenSpanDtype"):
            span_column = index
            with widget.widget_output:
                print(f"{column}")

    widget_components = [
        DataFrameTableComponent(widget=widget, dataframe=widget._df, update_metadata=update_metadata)
    ]

    if span_column != None:
        widget_components.append(DataFrameDocumentContainerComponent(dataframe=dataframe, span_column=span_column))

    return ipw.VBox(widget_components)

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

def DataFrameDocumentContainerComponent(dataframe, span_column=None, tag_column=None):
    """A Component that separates a dataframe by document and generates their components."""

    target_texts = []
    spans_by_text = {}

    if span_column != None:
        # Iterate over all rows in the frame and separate spans by target_text
        for index in range(len(dataframe["data"])):
            target_text = dataframe["data"][index][span_column].target_text
            if target_text not in target_texts:
                target_texts.append(target_text)
                spans_by_text[target_text] = []
            
            spans_by_text[target_text].append({
                "span": dataframe["data"][index][span_column],
                "index": index
            })

        # Generate a widget component for each document
        documents = []
        for text in target_texts:
            documents.append(DataFrameDocumentComponent(text=text, spans=spans_by_text[text]))
        
        return ipw.VBox(
            children=documents
        )

def DataFrameDocumentComponent(text, spans):
    """A component that renders the context of a document by generating visisble highlights for a column of spans."""

    # Create stacks
    begin_stack = []
    end_stack = []

    for entry in spans:
        span = entry["span"]
        index = entry["index"] # Index in the dataframe
        bisect.insort(begin_stack, (span.begin, index))
        bisect.insort(end_stack, (span.end, index))
    
    begin_stack = collections.deque(begin_stack)
    end_stack = collections.deque(end_stack)

    start_index = 0
    open_spans = []

    document_elements : list(str) = []

    
    while len(begin_stack) > 0 and len(end_stack) > 0:
        if(begin_stack[0][0] < end_stack[0][0]):
            if(len(open_spans) == 0):
                document_elements = document_elements + _get_linebreak_text_array(text[start_index:begin_stack[0][0]])

                start_index = begin_stack[0][0]
                # Add the span's ID to the open ID list
                open_spans.append(begin_stack[0][1])
                begin_stack.popleft()
            else:
                span_tag = "PL"
                span_text = text[start_index:begin_stack[0][0]]
                document_elements.append(DocumentSpan(text=text[start_index:begin_stack[0][0]], tag=span_tag, show_tag=False))

                start_index = begin_stack[0][0]
                # Add the span's ID to the open ID list
                open_spans.append(begin_stack[0][1])
                begin_stack.popleft()
        else:
            span_tag = "PL"
            span_text = text[start_index:end_stack[0][0]]
            document_elements.append(DocumentSpan(text=span_text, tag=span_tag, show_tag=True))

            start_index = end_stack[0][0]
            open_spans.remove(end_stack[0][1])
            end_stack.popleft()

    while len(end_stack) > 0:
        span_tag = "PL"
        span_text = text[start_index:end_stack[0][0]]
        document_elements.append(DocumentSpan(text=span_text, tag=span_tag, show_tag=True))

        start_index = end_stack[0][0]
        open_spans.remove(end_stack[0][1])
        end_stack.popleft()

    document_elements.append(text[start_index:])

    return ipw.HTML(
        f"""
            <div class='document'>
                {"".join(document_elements)}
            </div>
        """
    )

def DocumentSpan(text, tag, show_tag=True, bgcolor="rgba(200, 180, 255, 0.5)") -> str:
    return f"""
        <span style="line-height: 2; display: inline-block; padding: 0 0.2em; background-color: {bgcolor};">{text}</span>
    """

def _get_linebreak_text_array(in_text: str) -> str:
    splitarr = in_text.split('\n')
    i = 1
    while(i < len(splitarr)):
        splitarr.insert(i, "<br>")
        i += 2
    return splitarr