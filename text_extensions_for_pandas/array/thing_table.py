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
# thing_table.py
#
# Part of text_extensions_for_pandas
#
# Data structure for managing collections of immutable items that implement
# __hash__ and __eq__. Serves as a base class for StringTable
#
from abc import ABC, abstractmethod

import numpy as np
import textwrap
from typing import *


class ThingTable(ABC):
    """
    A set of immutable things, plus integer IDs for said things.

    Also implicitly maps `None` to ID -1.

    Serves as a base class for collections of specific things like strings and
    tokenizations.
    """

    # Special integer ID for None as a thing.
    NONE_ID = -1

    # Special integer ID for "not an id"
    NOT_AN_ID = -2

    def __init__(self):
        # Bidirectional map from unique string to integer ID and back
        self._thing_to_id = {}  # type: Dict[Any, int]
        self._id_to_thing = []  # type: List[Any]
        self._total_bytes = 0  # type: int

    def thing_to_id(self, thing: Any) -> int:
        """
        :param thing: A thing to look up in this table
        :returns: One of:
        * The integer ID of the indicated thing, if present.
        * `ThingTable.NONE_ID` if string is None
        * `ThingTable.NOT_AN_ID` if string is not present in the table
        """
        if thing is None:
            # By convention, None maps to -1
            return ThingTable.NONE_ID
        elif thing not in self._thing_to_id:
            return ThingTable.NOT_AN_ID
        else:
            return self._thing_to_id[thing]

    def id_to_thing(self, int_id: int) -> Any:
        """
        :param int_id: Integer ID that is potentially associated with a thing in the
         table
        :return: The associated thing, if present, or `None` if no thing is associated
         with the indicated ID.
        """
        if int_id <= ThingTable.NOT_AN_ID:
            raise ValueError(f"Invalid ID {int_id}")
        if ThingTable.NONE_ID == int_id:
            return None
        return self._id_to_thing[int_id]

    def add_thing(self, thing: Any) -> int:
        """
        Adds a thing to the table. Raises a ValueError if the thing is already
        present.

        :param thing: Thing to add
        :return: unique ID for this thing
        """
        if thing in self._thing_to_id:
            raise ValueError(f"'{textwrap.shorten(s, 40)}' already in table")
        new_id = len(self._id_to_thing)
        self._id_to_thing.append(thing)
        self._thing_to_id[thing] = new_id
        self._total_bytes += self.size_of_thing(thing)
        return new_id

    @abstractmethod
    def size_of_thing(self, thing: Any) -> int:
        """
        :param thing: Thing to be insterted in this table
        :return: The number of bytes that the thing occupies in memory
        """
        pass

    def maybe_add_thing(self, thing: Any) -> int:
        """
        Adds a thing to the table if it is not already present.

        :param thing: Thing to add
        :return: unique ID for this thing
        """
        if thing is not None and thing not in self._thing_to_id:
            return self.add_thing(thing)
        else:
            return self.thing_to_id(thing)

    def maybe_add_strs(self, s: Sequence[str]) -> np.ndarray:
        """
        Vectorized version of :func:`maybe_add_str` for translating, and
        potentially adding multiple strings at once.

        :param s: Multiple strings to be translated and potentially added
        :returns: A numpy array of the corresponding integer IDs for the strings.
        Adds each string to the table if it is not already present.
        """
        result = np.empty(len(s), dtype=np.int)
        for i in range(len(result)):
            result[i] = self.maybe_add_str(s[i])
        return result

    def nbytes(self):
        """
        Number of bytes in a (currently hypothetical) serialized version of this table.
        """
        return self._total_bytes

    def num_things(self):
        """
        :return: Number of distinct things in the table
        """
        return len(self._id_to_thing)





