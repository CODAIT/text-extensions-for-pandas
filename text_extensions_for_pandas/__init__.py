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
# NLP addons for Pandas DataFrames.
#
# To use:
#   import text_extensions_for_pandas as tp
#


# We expose the extension array classes at the top level of our namespace.
from text_extensions_for_pandas.array.span import (
    Span, SpanDtype, SpanArray
)
from text_extensions_for_pandas.array.token_span import (
    TokenSpan, TokenSpanDtype, TokenSpanArray
)
from text_extensions_for_pandas.array.tensor import (
    TensorElement, TensorDtype, TensorArray
)

# Import this file to activate our Pandas series accessor callbacks
import text_extensions_for_pandas.array.accessor

# Sub-modules
from text_extensions_for_pandas import io
from text_extensions_for_pandas import spanner

# Sphinx autodoc needs this redundant listing of public symbols to list the contents
# of this subpackage.
__all__ = [
    "Span", "SpanDtype", "SpanArray",
    "TokenSpan", "TokenSpanDtype", "TokenSpanArray",
    "TensorElement", "TensorDtype", "TensorArray",
    "io"
]