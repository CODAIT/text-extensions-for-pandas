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
import textwrap
from typing import *

class StringTable(object):
    """
    A set of immutable strings, plus integer IDs for said strings.

    Also implicitly maps `None` to ID -1.
    """

    # Special integer ID for None as a string.
    NONE_ID = -1

    # Special integer ID for "not an id"
    NOT_AN_ID = -2

    def __init__(self):
        # Bidirectional map from unique string to integer ID and back
        self._str_to_id = {}  # type: Dict[str, int]
        self._id_to_str = []  # type: List[str]
        self._total_bytes = 0  # type: int

    @classmethod
    def single_string_table(cls, s: str) -> "StringTable":
        """
        Factory method for building a StringTable containing a single string at ID 0.

        Users of this class are encouraged to use this method when possible,
        so that performance tuning can be localized to this method.
        """
        # For now we return a fresh table each time.
        ret = StringTable()
        ret.maybe_add_str(s)
        return ret

    @classmethod
    def merge_string_tables_and_ids(cls, tables: Sequence["StringTable"],
                                    int_ids: Sequence[np.ndarray]):
        """
        Factory method for combining together multiple references to different
        StringTables into references to a new, combined StringTable.

        Users of this class are encouraged to use this method when possible,
        so that performance tuning can be localized to this method.

        :param tables: A list of (possibly) different mappings from int to string
        :param int_ids: List of lists of integer IDs that decode to strings via the
         corresponding elements of `tables`.

        :returns: A tuple containing:
        * A new, merged StringTable containing all the unique strings under `tables`
          that are referenced in `int_ids` (and possibly additional strings that aren't
          referenced)
        * Numpy arrays of integer offsets into the new StringTable, corresponding to the
          elements of `int_ids`
        """
        if len(tables) != len(int_ids):
            raise ValueError(f"Got {len(tables)} StringTables "
                             f"and {len(int_ids)} lists of IDs.")
        new_table = StringTable()
        new_ids_list = []
        for i in range(len(tables)):
            old_table = tables[i]
            old_ids = int_ids[i]
            new_ids = np.empty_like(old_ids, dtype=int)
            old_id_to_new_id = [
                new_table.maybe_add_str(old_table._id_to_str[j])
                for j in range(len(old_table._id_to_str))
            ]
            for j in range(len(old_ids)):
                new_ids[j] = old_id_to_new_id[old_ids[j]]
            new_ids_list.append(new_ids)
        return new_table, new_ids_list

    @classmethod
    def merge_strings(cls, strings: Sequence[str]):
        """
        Factory method for bulk-adding multiple strings to create a single
        StringTable and a list of integer IDs against that StringTable.

        Users of this class are encouraged to use this method when possible,
        so that performance tuning can be localized to this method.

        :param strings: Strings to be de-duplicated and converted to a StringTable.
        :returns: Two values:
        * A StringTable containing (at least) all the unique strings in `strings`
        * A Numppy array of integer string IDs against the returned StringTable, where
          each ID maps to the corresponding element of `strings`
        """
        new_table = StringTable()
        str_ids = np.empty(len(strings), dtype=int)
        for i in range(len(strings)):
            str_ids[i] = new_table.maybe_add_str(strings[i])
        return new_table, str_ids

    def string_to_id(self, string: str) -> int:
        """
        :param string: A string to look up in this table
        :returns: One of:
        * The integer ID of the indicated string, if present.
        * `StringTable.NONE_ID` if string is None
        * `StringTable.NOT_AN_ID` if string is not present in the table
        """
        if string is None:
            # By convention, None maps to -1
            return StringTable.NONE_ID
        elif string not in self._str_to_id:
            return StringTable.NOT_AN_ID
        else:
            return self._str_to_id[string]

    def id_to_string(self, int_id: int) -> str:
        if int_id <= StringTable.NOT_AN_ID:
            raise ValueError(f"Invalid string ID {int_id}")
        if StringTable.NONE_ID == int_id:
            return None
        return self._id_to_str[int_id]

    def strings_to_ids(self, strings: Sequence[str]) -> np.ndarray:
        """
        Vectorized version of :func:`string_to_id` for translating multiple strings
        at once.

        :param strings: Multiple strings to be translated to IDs
        :returns: A numpy array of the same integers that :func:`string_to_id` would
         return.
        """
        ret = np.empty(len(strings), dtype=int)
        for i in range(len(strings)):
            ret[i] = self.string_to_id(strings[i])
        return ret

    def ids_to_strings(self, int_ids: Sequence[int]) -> np.ndarray:
        """
        Vectorized version of :func:`id_to_string` for translating multiple IDs
        at once.

        :param int_ids: Multiple integer IDs to be translated to strings
        :returns: A numpy array of string objects.
        """
        ret = np.empty(len(int_ids), dtype=object)
        for i in range(len(int_ids)):
            ret[i] = self.id_to_string(int_ids[i])
        return ret



    def add_str(self, s: str) -> int:
        """
        Adds a string to the table. Raises a ValueError if the string is already
        present.

        :param s: String to add
        :return: unique ID for this string
        """
        if s in self._str_to_id:
            raise ValueError(f"String '{textwrap.shorten(s, 40)}' already in table")
        new_id = len(self._id_to_str)
        self._id_to_str.append(s)
        self._str_to_id[s] = new_id
        self._total_bytes += len(s.encode("utf-8"))
        return new_id

    def maybe_add_str(self, s: str) -> int:
        """
        Adds a string to the table if it is not already present.

        :param s: String to add
        :return: unique ID for this string
        """
        if s is not None and s not in self._str_to_id:
            return self.add_str(s)
        else:
            return self.string_to_id(s)

    def nbytes(self):
        """
        Number of bytes in a (currently hypothetical) serialized version of this table.
        """
        return self._total_bytes





