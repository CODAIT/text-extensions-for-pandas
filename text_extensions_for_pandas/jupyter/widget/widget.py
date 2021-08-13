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

import pandas as pd
import ipywidgets as ipw
from IPython.display import display, clear_output, HTML
import text_extensions_for_pandas.jupyter.widget.span as tep_span
import text_extensions_for_pandas.jupyter.widget.table as tep_table

import text_extensions_for_pandas.resources

# TODO: This try/except block is for Python 3.6 support, and should be
# reduced to just importing importlib.resources when 3.6 support is dropped.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

_WIDGET_SCRIPT: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "dataframe_widget.js")
_WIDGET_STYLE: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "dataframe_widget.css")
_WIDGET_TABLE_CONVERT_SCRIPT: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "dataframe_widget_table_converter.js")

class DataFrameWidget():

    def __init__(self, dataframe, metadata_column=None, interactive_columns=None):
        """An instance of an interactive widget that will display Text Extension for Pandas types Span and TokenSpan in their document contexts beside a visualization of the backing dataframe.
        Provides interactive table elements, multiple Span coloring modes, and tools to analyze, modify, and extend DataFrame-backed datasets.
        
        params:
        dataframe (pandas.DataFrame): The DataFrame to visualize in the widget.
        metadata_column (pandas.Series, optional): Series of selected values to pre-load into the index column.
        interactive_columns (list, optional): List of column names to preset as interactive.
        """

        self._df = dataframe.copy(deep=True)

        # Refreshable Outputs
        self.widget_output = ipw.Output()
        self.debug_output = ipw.Output()
        self.widget_output.add_class("tep--dfwidget--output")
        self._document_output = None

        # Span Visualization Globals
        self._tag_display = None
        self._color_mode = 'ROW'

        # Initialize selected column
        if (metadata_column):
            md_length = len(metadata_column)
            # Check that metadata matches the length of the index. If too short or too long, mutate
            if (md_length < self._df.shape[0]):
                metadata_column = metadata_column + [False for i in range(md_length, self._df.shape[0])]
            elif (md_length > self._df.shape[0]):
                metadata_column = metadata_column[:self._df.shape[0]]
            # Now we have a full starting array to create a series
            self._metadata_column = pd.Series(metadata_column, index=self._df.index)
        else:
            self._metadata_column = pd.Series([False for i in range(self._df.shape[0])], index=self._df.index)


        # Initialize interactive columns
        self.interactive_columns = dict()
        for column in self._df.columns.values:
            self.interactive_columns[column] = False
        if(interactive_columns):
            for column in interactive_columns:
                self.interactive_columns.update({column: True})

        # Propagate initial values to components.
        self._update()

        # Attach the widget's script.
        with self.widget_output:
            display(HTML(f"<script>{_WIDGET_SCRIPT}</script>"))

    @property
    def selected(self) -> pd.Series:
        """A boolean series of the values of the selected rows in the table visualization."""
        return self._metadata_column

    def display(self) -> ipw.Widget:
        """Displays the widget. Returns a reference to the root output widget."""
        return self.widget_output
    
    def to_dataframe(self) -> pd.DataFrame:
        """Returns a copy of the DateFrame backing the internal state of the widget data.

        return:
        pandas.DataFrame: The copy of the backing dataframe.
        """
        return self._df.copy(deep=True)

    def set_interactive_columns(self, columns=[]):
        """Sets the columns to appear as interactive within the displayed widget.
            
        params:
        columns (list): A list of column names to appear as interactive.
        """
        # Reset the values
        self.interactive_columns = dict()
        for column in self._df.columns.values:
            self.interactive_columns[column] = False
        # Set the new values based on the parameter
        for column in columns:
            self.interactive_columns.update({column: True})


    # Internal methods to update or refresh widget state

    def _update(self):
        """Refresh the entire widget from scratch."""
        with self.widget_output:
            clear_output(wait=True)
            with self.debug_output:
                clear_output()
            display(self.debug_output)
            display(HTML(f"<script>{_WIDGET_TABLE_CONVERT_SCRIPT}</script>"))
            display(HTML(f"<style>{_WIDGET_STYLE}</style>"))
            display(ipw.VBox([DataFrameWidgetComponent(widget=self)]))

    def _update_document(self):
        """Only refresh the document display below the table."""
        if self._document_output:
            with self._document_output:
                clear_output(wait=True)
                display(tep_span.DataFrameDocumentContainerComponent(self))
    
    def _update_tag(self, change):
        """Updates the tag displayed on spans in the document view. Observe callback."""
        self._tag_display = change['new']
        self._update_document()

    def _update_color_mode(self, change):
        """Updates the color mode of span rendering. Observe callback."""
        self._color_mode = change['new']
        self._update_document()

    def _update_dataframe(self, value, column_name, column_index):
        """Updates the value at the indicated posiiton in the dataframe.
        
        params:
        value (any): The value to insert into the DataFrame.
        column_name (str): The name of the column to write to.
        column_index (int): The integer index within that column to write the value to.
        """
        self._df.at[column_index, column_name] = value

def DataFrameWidgetComponent(widget : DataFrameWidget) -> ipw.Widget:
    """The base component of the dataframe widget"""
    
    # Create the render with a table.
    widget_components = [
        tep_table.DataFrameTableComponent(widget=widget),
    ]

    # Try to generate a document. Will return NoneType if there are no spans to render.
    documents_widget = tep_span.DataFrameDocumentContainerComponent(widget=widget)
    if documents_widget:
        document_output = ipw.Output()
        document_output.add_class("tep--dfwidget--document-output")
        widget._document_output = document_output
        widget_components.append(document_output)
        with document_output:
            display(documents_widget)
    
    # Create and return a root widget node for all created components.
    root_widget = ipw.VBox(children=widget_components)
    root_widget.add_class("tep--dfwidget--root-container")

    return root_widget
