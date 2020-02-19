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
# pandas_text
#
# NLP addons for Pandas dataframes.
#
# To use:
#   import pandas_text as pt
#

# For now just expose everything at the top level of the namespace
from pandas_text.algebra import *
from pandas_text.char_span import *
from pandas_text.gremlin import *
from pandas_text.io import *
from pandas_text.span import *
from pandas_text.token_span import *

# Bring special Gremlin symbols up to the top-level scope
from pandas_text.gremlin import __
