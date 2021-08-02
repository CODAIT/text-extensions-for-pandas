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
import text_extensions_for_pandas.jupyter.widget.span as tep_span
import text_extensions_for_pandas.jupyter.widget.table as tep_table


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
        tep_table.DataFrameTableComponent(widget=widget, dataframe=widget._df, update_metadata=update_metadata)
    ]

    if span_column != None:
        widget_components.append(tep_span.DataFrameDocumentContainerComponent(dataframe=dataframe, span_column=span_column))

    return ipw.VBox(widget_components)
