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
The ``widget`` module contains the DataFrameWidget class and all supporting
functions for the use of the interactive DataFrame widget.
"""
################################################################################
# widget module
#
#
# Class and functions for the interactive DataFrame widget.

# Expose the public APIs that users should get from importing the top-level
# library.

from text_extensions_for_pandas.jupyter.widget.core import DataFrameWidget
