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

import numpy as np
import pandas as pd

from text_extensions_for_pandas.jupyter.widget.stubs import (
    ipw, display, clear_output, HTML)


def DataFrameTableComponent(widget) -> ipw.Widget:
    """Component representing the complete table of a dataframe widget."""
    # Avoids circular import
    from text_extensions_for_pandas.array.span import Span
    from text_extensions_for_pandas.array.token_span import TokenSpan

    # Locally-scoped change event handlers for the table

    def SpanOnBeginChange(change, span, text, column_name, index):
        new_span = Span(
            widget._df.get(column_name)[index].target_text, change["new"], span.end
        )
        widget._df.get(column_name)[index] = new_span
        text.value = str(new_span)
        widget._update_document()

    def SpanOnEndChange(change, span, text, column_name, index):
        new_span = Span(
            widget._df.get(column_name)[index].target_text, span.begin, change["new"]
        )
        widget._df.get(column_name)[index] = new_span
        text.value = str(new_span)
        widget._update_document()

    def TokenSpanOnBeginChange(
        change, token_span, text, column_name, index, data_begin, data_end
    ):
        new_span = TokenSpan(token_span.tokens, change["new"], token_span.end_token)
        widget._df.get(column_name)[index] = new_span
        text.value = str(new_span)

        data_begin.max = data_end.value - 1
        data_end.min = data_begin.value + 1
        widget._update_document()

    def TokenSpanOnEndChange(
        change, token_span, text, column_name, index, data_begin, data_end
    ):
        new_span = TokenSpan(token_span.tokens, token_span.begin_token, change["new"])
        widget._df.get(column_name)[index] = new_span
        text.value = str(new_span)

        data_begin.max = data_end.value - 1
        data_end.min = data_begin.value + 1
        widget._update_document()

    def MetaDataOnChange(change, index):
        widget._metadata_column[index] = change["new"]

    # Button click handlers

    def AddRowClickHandler(b):
        new_data = dataframe[len(dataframe) - 1 :]
        new_data.index = [len(dataframe)]
        widget._metadata_column = widget._metadata_column.append(
            pd.Series([False], index=[len(dataframe)])
        )
        widget._df = dataframe.append(new_data)
        widget._update()

    # Component rendering logic

    dataframe = widget._df

    table_columns = []

    # Generate the first column as the index + metadata controls box
    index_column = []
    index_column.append(ipw.HTML("<b>index</b>"))
    for index in dataframe.index:
        select_box = ipw.Checkbox(value=bool(widget._metadata_column[index]))
        select_box.observe(
            lambda change, index=index: MetaDataOnChange(change, index), names="value"
        )
        cell = ipw.HBox(
            [ipw.HTML(str(index)), select_box],
            layout=ipw.Layout(justify_content="flex-end", margin="0"),
        )
        index_column.append(cell)
    index_column_widget = ipw.VBox(index_column)
    index_column_widget.add_class("tep--dfwidget--table-index")
    table_columns.append(index_column_widget)

    # Generate the remaining columns as either static or interactive variants
    for column in dataframe.columns:
        is_interactive = widget.interactive_columns[column]
        table_columns.append(
            DataFrameTableColumnComponent(
                widget,
                is_interactive,
                dataframe[column],
                SpanOnBeginChange,
                SpanOnEndChange,
                TokenSpanOnBeginChange,
                TokenSpanOnEndChange,
            )
        )

    # After the table has been rendered, append any table controls below.
    button = ipw.Button(description="Add Row")
    button.on_click(AddRowClickHandler)

    # Finally, construct and return the widget structure.
    table = ipw.HBox(children=table_columns)
    table.add_class("tep--dfwidget--table")

    table_container = ipw.VBox(children=[table, button])
    table_container.add_class("tep--dfwidget--table-container")

    return table_container


def DataFrameTableColumnComponent(
    widget,
    is_interactive,
    column,
    SpanOnBeginChange,
    SpanOnEndChange,
    TokenSpanOnBeginChange,
    TokenSpanOnEndChange,
):
    """Constructs the widget component representing a single column of data within the
    table."""

    column_items = []

    # Begin with the column header.
    column_items.append(ipw.HTML(f"<b>{column.name}</b>"))

    # Branch to create either a static or interactive version of the column.
    if is_interactive:
        # Build an interactive column.
        for column_index in range(len(column)):
            # Call handler to handle columns of different data types
            # Checks if column is a categorical type and passes in the list of all possible categories as an additional argument if it is
            if str(column.dtype) == "category":
                data = DataTypeHandler(
                    column, column[column_index], categories=column.cat.categories
                )
                data.observe(
                    lambda change, column_index=column_index, column_name=column.name, widget=widget: (
                        CategoricalHandler(change, column_index, column_name, widget)
                    ),
                    names="value",
                )

            elif str(column.dtype) == "SpanDtype":
                data_begin, data_end, text = SpanHandler(
                    column, column_index, SpanOnBeginChange, SpanOnEndChange
                )

            elif str(column.dtype) == "TokenSpanDtype":
                data_begin, data_end, text = TokenSpanHandler(
                    column, column_index, TokenSpanOnBeginChange, TokenSpanOnEndChange
                )

            else:
                data = DataTypeHandler(column, column[column_index])
                data.observe(
                    lambda change, column_name=column.name, index=column_index, widget=widget: GenericInteractHandler(
                        widget, change["new"], column_name, index
                    ),
                    names="value",
                )

            # Create a single container for all widget components.
            if (
                str(column.dtype) == "SpanDtype"
                or str(column.dtype) == "TokenSpanDtype"
            ):
                widget_ui = ipw.HBox([data_begin, data_end, text])
            else:
                widget_ui = ipw.HBox([data])
            widget_ui.add_class("tep--dfwidget--table-widget")

            # Adds interactive widgets to table.
            cell_widget = ipw.HBox(
                children=[widget_ui],
                layout=ipw.Layout(justify_content="flex-end", margin="0"),
            )
            cell_widget.add_class(f"tep--dfwidget--row-{column_index}")

            column_items.append(cell_widget)
    else:
        # Build a static column
        for column_index in range(len(column)):
            item = column[column_index]

            cell_widget = ipw.HBox(
                children=[ipw.HTML(f"<div>{str(item)}</div>")],
                layout=ipw.Layout(justify_content="flex-end", margin="0"),
            )
            cell_widget.add_class(f"tep--dfwidget--row-{column_index}")
            column_items.append(cell_widget)

    return ipw.VBox(children=column_items, layout=ipw.Layout(border="0px solid black"))


# Respond to change events in generic widgets by updating that data in the parent widget's dataframe.
def GenericInteractHandler(widget, new_value, column_name, index):
    """Generic callback for interactive value updates."""
    widget.update_dataframe(new_value, column_name, index)
    return new_value


# Generates different interactive widget depending on data type of column
def DataTypeHandler(
    column, value, categories=None, min=None, max=None, description=None
):
    """Returns an interactive widget based on the type of objects in the column"""
    if pd.api.types.is_object_dtype(
        column
    ):  # < ! DANGEROUS. Will treat multi-typed columns as strings. Requires user to be responsible with column types.
        return ipw.Text(value=str(value))
    elif pd.api.types.is_integer_dtype(column):
        return ipw.IntText(value=value)
    elif pd.api.types.is_float_dtype(column):
        return ipw.FloatText(value=value)
    elif pd.api.types.is_bool_dtype(column):
        return ipw.Checkbox(value=bool(value))
    elif pd.api.types.is_categorical_dtype(column):
        if value == np.nan or value == None or str(value) == "nan":
            value = "nan"
        return ipw.Dropdown(options=["nan", *categories.array], value=value)
    # used for handling spans
    elif str(column.dtype) == "SpanDtype" or str(column.dtype) == "TokenSpanDtype":
        return ipw.BoundedIntText(
            value=value, min=min, max=max, description=description
        )
    else:
        return ipw.Text(value=str(value), disabled=True)


# Creates widgets for taking in int inputs to change span.begin and span.end and an output text widget
def SpanHandler(column, column_index, on_begin_change, on_end_change):
    """Creates a custom widget for the Text Extensions for Pandas Span object type."""
    # Create input bounded int boxes
    data_begin = DataTypeHandler(
        column,
        column[column_index].begin,
        min=0,
        max=len(column[column_index]._text),
        description="Begin: ",
    )

    data_end = DataTypeHandler(
        column,
        column[column_index].end,
        min=0,
        max=len(column[column_index]._text),
        description="End: ",
    )

    # Create output text box
    text = ipw.Text(str(column[column_index]), disabled=True)

    # Trigger functions on changing values in input int boxes
    data_begin.observe(
        lambda change, i=column_index, text=text, column_name=column.name: on_begin_change(
            change, column[i], text, column_name, i
        ),
        names="value",
    )

    data_end.observe(
        lambda change, i=column_index, text=text, column_name=column.name: on_end_change(
            change, column[i], text, column_name, i
        ),
        names="value",
    )

    # Add some identifying classed
    data_begin.add_class("tep--dfwidget--span-widget")
    data_end.add_class("tep--dfwidget--span-widget")

    return data_begin, data_end, text


def TokenSpanHandler(
    column, column_index, TokenSpanOnBeginChange, TokenSpanOnEndChange
):
    """Creates a custom widget for the Text Extensions for Pandas TokenSpan object type."""
    # Create input bounded int boxes
    data_begin = DataTypeHandler(
        column,
        column[column_index].begin_token,
        min=0,
        max=column[column_index].end_token,
        description="Begin Token: ",
    )

    data_end = DataTypeHandler(
        column,
        column[column_index].end_token,
        min=column[column_index].begin_token,
        max=len(column[column_index].tokens),
        description="End Token: ",
    )

    # Create output text box
    text = ipw.Text(str(column[column_index]), disabled=True)

    # Trigger functions on changing values in input int boxes
    data_begin.observe(
        lambda change, index=column_index, text=text, column_name=column.name: TokenSpanOnBeginChange(
            change, column[index], text, column_name, index, data_begin, data_end
        ),
        names="value",
    )

    data_end.observe(
        lambda change, index=column_index, text=text, column_name=column.name: TokenSpanOnEndChange(
            change, column[index], text, column_name, index, data_begin, data_end
        ),
        names="value",
    )

    # Add some identifying classed
    data_begin.add_class("tep--dfwidget--span-widget")
    data_end.add_class("tep--dfwidget--span-widget")

    return data_begin, data_end, text


def CategoricalHandler(change, column_index, column_name, widget):
    """Event callback for interactive changes in a categorical column. Necessary due
    to the use of np.NaN for a None type."""
    if change["new"] == "nan":
        widget._df.get(column_name)[column_index] = np.nan
    else:
        widget._df.get(column_name)[column_index] = change["new"]
    widget._update_document()
