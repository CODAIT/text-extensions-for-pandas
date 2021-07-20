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

def render(dataframe):
    
    # This import ensures proper idomwidget hooks are invoked
    import idom_jupyter

    return DataFrameWidget({
        "dataframe": dataframe
    })

@idom.component
def DataFrameWidget(props):
    """The base component of the dataframe widget"""
    return idom.html.div(
        "Placeholder"
    )