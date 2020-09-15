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

"""
The ``spanner`` module of Text Extensions for Pandas provides span-specific operations
for Pandas DataFrames, based on the Document Spanners formalism, also known as
spanner algebra.

Spanner algebra is an extension of relational algebra with additional operations
to cover NLP applications. See the paper ["Document Spanners: A Formal Approach to
Information Extraction"](
https://researcher.watson.ibm.com/researcher/files/us-fagin/jacm15.pdf) by Fagin et al.
for more information.
"""

# The contents of this package are divided into multiple Python source files, but we
# export the symbols at the root of the package namespace.
from text_extensions_for_pandas.spanner.consolidate import (
    consolidate
)
from text_extensions_for_pandas.spanner.extract import (
    extract_dict,
    extract_regex_tok
)
from text_extensions_for_pandas.spanner.join import (
    adjacent_join,
    contain_join,
    overlap_join
)
from text_extensions_for_pandas.spanner.project import (
    lemmatize
)

# Sphinx autodoc needs this redundant listing of public functions to list the contents
# of this subpackage.
__all__ = [
    # consolidate.py
    "consolidate",

    # extract.py
    "extract_dict",
    "extract_regex_tok",

    # join.py
    "adjacent_join",
    "contain_join",
    "overlap_join",
    
    # project.py
    "lemmatize"
]



