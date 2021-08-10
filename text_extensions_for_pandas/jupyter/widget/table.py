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

from IPython.core.display import clear_output
import ipywidgets as ipw
import numpy as np
import pandas as pd

def DataFrameTableComponent(widget, dataframe, update_metadata):
    """Component representing the complete table of a dataframe widget."""
    #Avoids circular import 
    from text_extensions_for_pandas.array.span import Span
    from text_extensions_for_pandas.array.token_span import TokenSpan

    def InteractHandler(data, column_name, index):
        widget.update_dataframe(data, column_name, index)
        return data
    
    #Updates span widgets on cchange in either the begin int input box or the end int input box
    def SpanOnBeginChange(change, span, text, column_name, index):
        with text:
            clear_output(wait = True)
            widget._df.get(column_name)[index] = Span(widget._df.get(column_name)[index].target_text, change['new'], span.end)
            print(span.target_text[change['new']: span.end])
        widget._update_document()
    
    def SpanOnEndChange(change, span, text, column_name, index):
        with text:
            clear_output(wait = True)
            widget._df.get(column_name)[index] = Span(widget._df.get(column_name)[index].target_text, span.begin, change['new'])
            print(span.target_text[span.begin: change['new']])
        widget._update_document()

    def TokenSpanOnBeginChange(change, token_span, text, column_name, index, data_begin, data_end):
        with text:
            clear_output(wait = True)
            widget._df.get(column_name)[index] = TokenSpan(token_span.tokens, change['new'], token_span.end_token)
            for token in token_span.tokens[change['new']: token_span.end_token]:
                print(token.covered_text)

        data_begin.max = data_end.value - 1
        data_end.min = data_begin.value + 1
        widget._update_document()
    def TokenSpanOnEndChange(change, token_span, text, column_name, index, data_begin, data_end):
        with text:
            clear_output(wait = True)
            widget._df.get(column_name)[index] = TokenSpan(token_span.tokens, token_span.begin_token, change['new'])
            for token in token_span.tokens[token_span.begin_token : change['new']]:
                print(token.covered_text)
        data_begin.max = data_end.value - 1
        data_end.min = data_begin.value + 1
        widget._update_document()

    def on_metadata_change(change, index):
        widget._metadata_column[index] = change["new"]

    table_columns = []
    # First column is an index/select box column
    index_column = []
    index_column.append(ipw.HTML("<b>index</b>"))
    for index in dataframe.index:
        select_box = ipw.Checkbox(value=bool(widget._metadata_column[index]))
        select_box.observe(lambda change, index=index: on_metadata_change(change, index), names='value')
        cell = ipw.HBox([ipw.HTML(str(index)), select_box], layout=ipw.Layout(justify_content="flex-end", border='1px solid gray', margin='0'))
        index_column.append(cell)
    index_column_widget = ipw.VBox(index_column)
    index_column_widget.add_class("tep--dfwidget--table-index")
    table_columns.append(index_column_widget)

    for column in dataframe.columns:
        is_selected = widget.selected_columns[column]

        table_columns.append(DataFrameTableColumnComponent(widget, is_selected, dataframe[column], InteractHandler, SpanOnBeginChange, SpanOnEndChange, TokenSpanOnBeginChange, TokenSpanOnEndChange))

    button = ipw.Button(description="Add Row")

    def AddRow(b):
        new_data = dataframe[len(dataframe)-1:]
        new_data.index = [len(dataframe)]
        widget._metadata_column = widget._metadata_column.append(pd.Series([False], index=[len(dataframe)]))
        widget._df = dataframe.append(new_data)
        widget._update()
    
    button.on_click(AddRow)

    table = ipw.HBox(children=table_columns)
    table.add_class("tep--dfwidget--table")

    table_container = ipw.VBox(children = [table, button])
    table_container.add_class("tep--dfwidget--table-container")
    
    return table_container

def DataFrameTableColumnComponent(widget, is_selected, column, InteractHandler, SpanOnBeginChange, SpanOnEndChange, TokenSpanOnBeginChange, TokenSpanOnEndChange):

    column_items = []
    # Column Header
    column_items.append(
        ipw.HTML(f"<b>{column.name}</b>")
    )
    #Create an interactive ipywidget 
    if(is_selected):
        for column_index in range(len(column)):
             #Call handler to handle columns of different data types
             #Checks if column is a categorical type and passes in the list of all possible categories as an additional argument if it is
            if(str(column.dtype) == "category"):
                data = DataTypeHandler("category", column[column_index], categories = column.cat.categories)
                data.observe(
                    lambda change, column_index=column_index, column_name=column.name, widget=widget: (
                        CategoricalHandler(change, column_index, column_name, widget)),
                    names='value')
            
            elif(str(column.dtype) == "SpanDtype"):

                data_begin, data_end, text = SpanHandler(column, column_index, SpanOnBeginChange, SpanOnEndChange) 
            elif(str(column.dtype) == "TokenSpanDtype"):
                data_begin, data_end, text = TokenSpanHandler(column, column_index, TokenSpanOnBeginChange, TokenSpanOnEndChange) 

            else:
                data = DataTypeHandler(column.dtypes, column[column_index])
            
            column_name = ipw.Text(value = column.name, disabled = True)           

            index = ipw.IntText(value = column_index, disabled = True)

            if(str(column.dtype) == "SpanDtype" or str(column.dtype) == "TokenSpanDtype"):
                widget_ui = ipw.HBox([data_begin, data_end, text])  
            else:
                widget_ui = ipw.HBox([data])  
            
            #Column name and index are passed into InteractHandler as widget components that are not rendered
            if(str(column.dtype) != "SpanDtype" and str(column.dtype) != "TokenSpanDtype" and str(column.dtype) != "category"):
                interactiveWidget = ipw.interactive_output(InteractHandler, {'data': data, 'column_name' : column_name, "index" : index})
            
            #Adds interactive widgets to table 
            cell_widget = ipw.HBox(children=[widget_ui], layout=ipw.Layout(justify_content="flex-end", border='1px solid gray', margin='0'))
            cell_widget.add_class(f"tep--dfwidget--row-{column_index}")

            column_items.append(cell_widget)
    else:
    # Column Items
        for column_index in range(len(column)):
            item = column[column_index]

            cell_widget = ipw.HBox(children=[ipw.HTML(f"<div>{str(item)}</div>")], layout=ipw.Layout(justify_content="flex-end", border='1px solid gray', margin='0'))
            cell_widget.add_class(f"tep--dfwidget--row-{column_index}")
            column_items.append(cell_widget)
    
    return ipw.VBox(children=column_items, layout=ipw.Layout(border='0px solid black'))
#Generates different interactive widget depending on data type of column    
def DataTypeHandler(datatype, value, categories = None, min = None, max = None, description = None):
    if(str(datatype) == 'object'):
        return ipw.Text(value = str(value))
    elif(str(datatype) == 'int64'):
        return ipw.IntText(value = value)
    #used for handling spans
    elif(str(datatype) == 'bounded_int64'):
        return ipw.BoundedIntText(
            value = value,
            min = min,
            max = max,
            description = description
        )
    elif(str(datatype) == 'float64'):
        return ipw.FloatText(value = value)
    elif(str(datatype) == 'bool'):
        return ipw.Checkbox(value = bool(value))
    elif(str(datatype) == 'category'):
        if value == np.nan or value == None or str(value) == 'nan':
            value = 'nan'
        return ipw.Dropdown(
            options = ['nan', *categories.array],
            value = value)

#Creates widgets for taking in int inputs to change span.begin and span.end and an output text widget
def SpanHandler(column, column_index, on_begin_change, on_end_change):
    #Create input bounded int boxes
    data_begin = DataTypeHandler('bounded_int64', 
        column[column_index].begin, 
        min = 0, 
        max = len(column[column_index]._text), 
        description = "Begin: ")

    data_end = DataTypeHandler('bounded_int64', 
        column[column_index].end,
        min = 0, 
        max = len(column[column_index]._text), 
        description = "End: ")

    #Create output text box
    text = ipw.Output()
    with text:
        print(column[column_index].target_text[column[column_index].begin : column[column_index].end])

    #Trigger functions on changing values in input int boxes
    data_begin.observe(
        lambda change, 
        i = column_index, 
        text = text, 
        column_name = column.name : on_begin_change(change, column[i], text, column_name, i), 
        names='value')
    
    data_end.observe(
        lambda change, 
        i = column_index, 
        text = text, 
        column_name = column.name : on_end_change(change, column[i], text, column_name, i), 
        names='value') 
    
    return data_begin, data_end, text
    
def TokenSpanHandler(column, column_index, TokenSpanOnBeginChange, TokenSpanOnEndChange):
    #Create input bounded int boxes
    data_begin = DataTypeHandler('bounded_int64', 
        column[column_index].begin_token, 
        min = 0, 
        max = column[column_index].end_token, 
        description = "Begin Token: ")

    data_end = DataTypeHandler('bounded_int64', 
        column[column_index].end_token, 
        min = column[column_index].begin_token, 
        max = len(column[column_index].tokens), 
        description = "End Token: ")

    #Create output text box
    text = ipw.Output()
    with text:
        for token in column[column_index].tokens[column[column_index].begin_token : column[column_index].end_token]:
            print(token.covered_text)

    #Trigger functions on changing values in input int boxes
    data_begin.observe(
        lambda change, 
        index = column_index, 
        text = text, 
        column_name = column.name : TokenSpanOnBeginChange(change, column[index], text, column_name, index, data_begin, data_end), 
        names='value')

    data_end.observe(
        lambda change, 
        index = column_index, 
        text = text, column_name = column.name : TokenSpanOnEndChange(change, column[index], text, column_name, index, data_begin, data_end), 
        names='value') 

    return data_begin, data_end, text

def CategoricalHandler(change, column_index, column_name, widget):
    if change["new"] == 'nan':
        widget._df.get(column_name)[column_index] = np.nan
    else:
        widget._df.get(column_name)[column_index] = change["new"]