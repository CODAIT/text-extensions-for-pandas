#
#  Copyright (c) 2020 IBM Corp.
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

################################################################################
# text_extensions_for_pandas
#
# NLP addons for Pandas dataframes.
#
# To use:
#   import text_extensions_for_pandas as pt
#

# For now just expose everything at the top level of the namespace
from text_extensions_for_pandas.spanner import *
from text_extensions_for_pandas.array import *
from text_extensions_for_pandas.io import *
from text_extensions_for_pandas.jupyter import *
