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
# token_table.py
#
# Part of text_extensions_for_pandas
#
# Data structures for managing collections of tokens. Used internally
# by span array types to store target document texts and their tokenizations.
#

from typing import *

from text_extensions_for_pandas import SpanArray
from text_extensions_for_pandas.array.thing_table import ThingTable


class _BoxedSpanArray:
    """
    Adapter around the SpanArray class that makes it compatible with Python
    dictionaries.
    """
    def __init__(self, tokens: SpanArray):
        self._tokens = tokens

    @property
    def tokens(self):
        return self._tokens

    def __hash__(self):
        return self._tokens.__hash__()

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, _BoxedSpanArray):
            return False
        else:
            # Built-in equals() function that has the same semantics as __eq__
            # is supposed to have
            return self.tokens.equals(other.tokens)

    def __ne__(self, other):
        return not self.__eq__(other)


class TokenTable(ThingTable):
    """
    A table of unique tokenizations of different documents.

    Each tokenization is represented with a :class:`SpanArray`. Because the
    `__eq__()` method of `SpanArray` returns a mask (for compatibility with
    Pandas), we need to wrap these SpanArray objects in an adapter to make
    them hashable.
    """

    def type_of_thing(self) -> Type:
        return SpanArray

    def size_of_thing(self, thing: Any) -> int:
        if not isinstance(thing, SpanArray):
            raise TypeError("TokenTable only works with SpanArrays.")
        return thing.nbytes

    def box(self, thing: Any) -> Any:
        return _BoxedSpanArray(thing)

    def unbox(self, boxed_thing: Any) -> Any:
        if not isinstance(boxed_thing, _BoxedSpanArray):
            raise TypeError("We didn't box this object")
        return boxed_thing.tokens
