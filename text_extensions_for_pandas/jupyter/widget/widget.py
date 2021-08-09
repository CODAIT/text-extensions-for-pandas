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

from IPython.core.display import clear_output
import pandas as pd
import ipywidgets as ipw
from pandas.core.frame import DataFrame
from IPython.display import display, Javascript, HTML
from traitlets import HasTraits, Dict, Int, Bool, All, default, observe
import text_extensions_for_pandas.jupyter.widget.span as tep_span
import text_extensions_for_pandas.jupyter.widget.table as tep_table

import text_extensions_for_pandas.resources

# TODO: This try/except block is for Python 3.6 support, and should be
# reduced to just importing importlib.resources when 3.6 support is dropped.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

_DEBUG_PRINTING = False
_WIDGET_SCRIPT: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "dataframe_widget.js")
_WIDGET_STYLE: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "dataframe_widget.css")


def render(dataframe, **kwargs):
    """Creates an instance of a DataFrame widget."""
    return DataFrameWidget(dataframe=dataframe, **kwargs)

class DataFrameWidget(HasTraits):

    _dataframe_dict = Dict()
    _dtypes = None
    widget = None
    widget_output = None
    debug_output = None

    def __init__(self, dataframe, metadata_column=None, selected_columns=None):

        self._df = dataframe.copy(deep=True)
        self._dataframe_dict = dataframe.to_dict("split")
        self._dtypes = dataframe.dtypes

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

        #Initialize selected_columns
        self.selected_columns = dict()
        for column in self._df.columns.values:
            self.selected_columns[column] = False
        if(selected_columns):
            self.selected_columns.update(selected_columns)

        # Initialize Widget        
        self.widget = DataFrameWidgetComponent(widget=self, update_metadata=self.update_metadata)
        self.widget.observe(self.print_change, names=All)

        # Display widget on root output
        with self.widget_output:
            display(self.debug_output)
            display(HTML(f"<style>{_WIDGET_STYLE}</style>"))
            display(ipw.VBox([self.widget]))
            display(HTML(f"<script>{_WIDGET_SCRIPT}</script>"))

    def display(self):
        """Displays the widget. Returns a reference to the root output widget."""
        return self.widget_output
    
    def _update(self):
        """Refresh the entire widget from scratch."""
        with self.widget_output:
            clear_output(wait=True)
            with self.debug_output:
                clear_output()
            display(self.debug_output)
            self.widget = DataFrameWidgetComponent(widget=self, update_metadata=self.update_metadata)
            self.widget.observe(self.print_change, names=All)
            display(HTML(f"<style>{_WIDGET_STYLE}</style>"))
            display(ipw.VBox([self.widget]))
    
    def _update_document(self):
        """Only refresh the document display below the table."""
        with self._document_output:
            clear_output(wait=True)
            display(tep_span.DataFrameDocumentContainerComponent(self, self._df))

    def to_dataframe(self):
        return self._df.copy(deep=True)

    def update_metadata(self, change):
        index = int(change['owner']._dom_classes[0])
        self._dataframe_dict["data"][index][0]["selected"] = change["new"]

    def print_change(self, change):
        """Prints the change event to the root output widget. Useful for change callback information."""
        if _DEBUG_PRINTING:
            with self.widget_output:
                print(change)

    def update_dataframe(self, value, column_name, column_index):
        """Updates the dataframe on interactive input. Interact callback."""
        self._df.at[column_index, column_name] = value
    
    def _update_tag(self, change):
        """Updates the tag displayed on spans in the document view. Observe callback."""
        self._tag_display = change['new']
        self._update_document()

    def _update_color_mode(self, change):
        """Updates the color mode of span rendering. Observe callback."""
        self._color_mode = change['new']
        self._update_document()

def DataFrameWidgetComponent(widget, update_metadata):
    """The base component of the dataframe widget"""
    
    # Create the render with a table.
    widget_components = [
        tep_table.DataFrameTableComponent(widget=widget, dataframe=widget._df, update_metadata=update_metadata),
    ]

    # Try to generate a document. Will return NoneType if there are no spans to render.
    documents_widget = tep_span.DataFrameDocumentContainerComponent(widget=widget, dataframe=widget._df)
    if documents_widget:
        document_output = ipw.Output()
        widget._document_output = document_output
        widget_components.append(document_output)
        with document_output:
            display(documents_widget)
    
    # Create and return a root widget node for all created components.
    root_widget = ipw.VBox(children=widget_components)

    return root_widget
