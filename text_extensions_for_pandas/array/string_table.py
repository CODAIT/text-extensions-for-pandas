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
# string_table.py
#
# Part of text_extensions_for_pandas
#
# Data structures for managing collections of immutable strings. Used internally
# by span array types to store target document texts.
#

import numpy as np

from typing import *

from text_extensions_for_pandas.array.thing_table import ThingTable


class StringTable(ThingTable):
    """
    A set of immutable strings, plus integer IDs for said strings.

    Also implicitly maps `None` to ID -1.
    """

    def type_of_thing(self) -> Type:
        return str

    def size_of_thing(self, thing: Any) -> int:
        if not isinstance(thing, str):
            raise TypeError(f"Only know how to handle strings; got {thing}")
        return len(thing.encode("utf-8"))






