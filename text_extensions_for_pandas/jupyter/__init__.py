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

"""
The ``jupyter`` module contains functions to support the use of Text Extensions for Pandas
in Jupyter notebooks.
"""
################################################################################
# jupyter module
#
#
# Functions in text_extensions_for_pandas for Jupyter notebook support.

# Expose the public APIs that users should get from importing the top-level
# library.
from text_extensions_for_pandas.jupyter.span import pretty_print_html
from text_extensions_for_pandas.jupyter.misc import run_with_progress_bar
from text_extensions_for_pandas.jupyter.widget import DataFrameWidget

__all__ = ["pretty_print_html", "run_with_progress_bar", "DataFrameWidget"]