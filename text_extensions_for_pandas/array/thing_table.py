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

import textwrap
#
# thing_table.py
#
# Part of text_extensions_for_pandas
#
# Data structure for managing collections of immutable items that implement
# __hash__ and __eq__. Serves as a base class for StringTable
#
from abc import ABC, abstractmethod
from typing import *

import numpy as np


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
        # Bidirectional map from unique thing (possibly boxed for dictionary
        # compatibility) to integer ID and back
        self._boxed_thing_to_id = {}  # type: Dict[Any, int]
        self._id_to_boxed_thing = []  # type: List[Any]
        self._total_bytes = 0  # type: int

    @abstractmethod
    def size_of_thing(self, thing: Any) -> int:
        """
        :param thing: Thing to be insterted in this table
        :return: The number of bytes that the thing occupies in memory
        """
        pass

    @abstractmethod
    def type_of_thing(self) -> Type:
        """
        :return: Expected type of things that this table will manage
        """
        pass

    def box(self, thing: Any) -> Any:
        """
        Subclasses should override this method if they manage items that aren't
        compatible with Python dictionaries.

        :param thing: Thing to insert into the table
        :return: a dictionary-compatible boxed version of `thing`, if such boxing
         is needed to make `thing` dictionary-compatible.
        """
        # Default implementation is a no-op
        return thing

    def unbox(self, boxed_thing: Any) -> Any:
        """
        Subclasses should override this method if they manage items that aren't
        compatible with Python dictionaries.

        :param boxed_thing: Thing that was boxed by this class's `box` method.
        :return: Original thing that was passed to `box`
        """
        # Default implementation is a no-op
        return boxed_thing

    @classmethod
    def create_single(cls, thing: Any):
        """
        Factory method for building a table containing a single value at ID 0.

        Users of this class are encouraged to use this method when possible,
        so that performance tuning can be localized to this method.
        """
        # For now we return a fresh table each time.
        ret = cls()
        ret.maybe_add_thing(thing)
        return ret

    @classmethod
    def merge_tables_and_ids(cls, tables: Sequence["ThingTable"],
                             int_ids: Sequence[np.ndarray]) \
            -> Tuple["ThingTable", np.ndarray]:
        """
        Factory method for combining together multiple references to different
        ThingTables into references to a new, combined ThingTable of the same type.

        Users of this class are encouraged to use this method when possible,
        so that performance tuning can be localized to this method.

        :param tables: A list of (possibly) different mappings from int to string
        :param int_ids: List of lists of integer IDs that decode to strings via the
         corresponding elements of `tables`.

        :returns: A tuple containing:
        * A new, merged table containing all the unique things under `tables`
          that are referenced in `int_ids` (and possibly additional things that aren't
          referenced)
        * Numpy arrays of integer offsets into the new table, corresponding to the
          elements of `int_ids`
        """
        if len(tables) != len(int_ids):
            raise ValueError(f"Got {len(tables)} {cls}s "
                             f"and {len(int_ids)} lists of IDs.")
        # TODO: Add fast-path code here to pass through the first table if
        #  both input tables are identical.
        new_table = cls()
        new_ids_list = []
        for i in range(len(tables)):
            old_table = tables[i]
            if not isinstance(old_table, cls):
                raise TypeError(f"Expected table of type {cls}, but got "
                                f"{type(old_table)}")
            old_ids = int_ids[i]
            if len(old_ids.shape) != 1:
                raise ValueError(f"Invalid shape for IDs {old_ids}")
            new_ids = np.empty_like(old_ids, dtype=int)
            old_id_to_new_id = [
                new_table.maybe_add_thing(old_table.id_to_thing(j))
                for j in range(old_table.num_things)
            ]
            for j in range(len(old_ids)):
                new_ids[j] = old_id_to_new_id[old_ids[j]]
            new_ids_list.append(new_ids)
        return new_table, new_ids_list

    @classmethod
    def merge_things(cls, things: Union[Sequence[Any], np.ndarray]):
        f"""
        Factory method for bulk-adding multiple things to create a single
        ThingTable and a list of integer IDs against that ThingTable.

        Users of this class are encouraged to use this method when possible,
        so that performance tuning can be localized to this method.

        :param things: things to be de-duplicated and converted to a ThingTable.
        :returns: Two values:
        * A ThingTable containing (at least) all the unique strings in `strings`
        * A Numppy array of integer string IDs against the returned ThingTable, where
          each ID maps to the corresponding element of `strings`
        """
        new_table = cls()
        str_ids = np.empty(len(things), dtype=int)
        for i in range(len(things)):
            str_ids[i] = new_table.maybe_add_thing(things[i])
        return new_table, str_ids

    @classmethod
    def from_things(cls, things: Union[Sequence[Any], np.ndarray]):
        """
        Factory method for creating a ThingTable from a sequence of unique things.

        :param things: sequence of unique things to be added to the ThingTable.
        :return: A ThingTable containing the elements of `things`.
        """
        new_table = cls()
        for thing in things:
            new_table.add_thing(thing)
        return new_table

    def thing_to_id(self, thing: Any) -> int:
        """
        :param thing: A thing to look up in this table
        :returns: One of:
        * The integer ID of the indicated thing, if present.
        * `ThingTable.NONE_ID` if thing is None
        * `ThingTable.NOT_AN_ID` if thing is not present in the table
        """
        if thing is None:
            # By convention, None maps to -1
            return ThingTable.NONE_ID
        elif not isinstance(thing, self.type_of_thing()):
            raise TypeError(f"Expected an object of type {self.type_of_thing()}, "
                            f"but received an object of type {type(thing)}")
        else:
            # Remaining branches require boxing for dictionary lookup
            boxed_thing = self.box(thing)
            if boxed_thing not in self._boxed_thing_to_id:
                return ThingTable.NOT_AN_ID
            else:
                return self._boxed_thing_to_id[boxed_thing]

    def id_to_thing(self, int_id: Union[int, np.int64, np.int32]) -> Any:
        """
        :param int_id: Integer ID that is potentially associated with a thing in the
         table
        :return: The associated thing, if present, or `None` if no thing is associated
         with the indicated ID.
        """
        if not isinstance(int_id, (int, np.int64, np.int32)):
            raise TypeError(f"Expected integer, but received {int_id} "
                            f"of type {type(int_id)}")
        elif int_id <= ThingTable.NOT_AN_ID:
            raise ValueError(f"Invalid ID {int_id}")
        elif ThingTable.NONE_ID == int_id:
            return None
        else:
            boxed_thing = self._id_to_boxed_thing[int_id]
            return self.unbox(boxed_thing)

    def ids_to_things(self, int_ids: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """
        Vectorized version of :func:`id_to_string` for translating multiple IDs
        at once.

        :param int_ids: Multiple integer IDs to be translated to strings
        :returns: A numpy array of string objects.
        """
        if not isinstance(int_ids, np.ndarray):
            int_ids = np.array(int_ids, dtype=int)
        if len(int_ids.shape) != 1:
            raise TypeError(f"Invalid shape {int_ids.shape} for array of integer IDs.")
        ret = np.empty(len(int_ids), dtype=object)
        for i in range(len(int_ids)):
            ret[i] = self.id_to_thing(int_ids[i].item())
        return ret

    def add_thing(self, thing: Any) -> int:
        """
        Adds a thing to the table. Raises a ValueError if the thing is already
        present.

        :param thing: Thing to add
        :return: unique ID for this thing
        """
        if not isinstance(thing, self.type_of_thing()):
            raise TypeError(f"Expected an object of type {self.type_of_thing()}, "
                            f"but received an object of type {type(thing)}")

        # Box for dictionary compatibility
        boxed_thing = self.box(thing)
        if boxed_thing in self._boxed_thing_to_id:
            raise ValueError(f"'{textwrap.shorten(str(thing), 40)}' already in table")
        new_id = len(self._id_to_boxed_thing)
        self._id_to_boxed_thing.append(boxed_thing)
        self._boxed_thing_to_id[boxed_thing] = new_id
        self._total_bytes += self.size_of_thing(thing)
        return new_id

    def maybe_add_thing(self, thing: Any) -> int:
        """
        Adds a thing to the table if it is not already present.

        :param thing: Thing to add
        :return: unique ID for this thing
        """
        if not isinstance(thing, self.type_of_thing()):
            raise TypeError(f"Expected an object of type {self.type_of_thing()}, "
                            f"but received an object of type {type(thing)}")

        current_id = self.thing_to_id(thing)
        if current_id != ThingTable.NOT_AN_ID:
            return current_id
        else:
            return self.add_thing(thing)

    def maybe_add_things(self, s: Sequence[Any]) -> np.ndarray:
        """
        Vectorized version of :func:`maybe_add_thing` for translating, and
        potentially adding multiple things at once.

        :param s: Multiple things to be translated and potentially added
        :returns: A numpy array of the corresponding integer IDs for the things.
        Adds each things to the table if it is not already present.
        """
        result = np.empty(len(s), dtype=np.int32)
        for i in range(len(result)):
            result[i] = self.maybe_add_thing(s[i])
        return result

    def nbytes(self):
        """
        Number of bytes in a (currently hypothetical) serialized version of this table.
        """
        return self._total_bytes

    @property
    def num_things(self) -> int:
        """
        :return: Number of distinct things in the table
        """
        return len(self._id_to_boxed_thing)

    @property
    def things(self) -> Iterator[Any]:
        """
        :return: Iterator over the unique things stored in this table.
        """
        return (self.unbox(thing) for thing in self._id_to_boxed_thing)

    @property
    def ids(self) -> Iterator[int]:
        """
        :return: Iterator over the IDs of things stored in this table, including the
         implicit ID ThingTable.NONE_ID
        """
        if ThingTable.NONE_ID != -1:
            raise ValueError("Someone has changed the value of NONE_ID; need to rewrite "
                             "this function.")
        return range(-1, len(self._id_to_boxed_thing))

    def things_to_ids(self, things: Sequence[Any]) -> np.ndarray:
        """
        Vectorized version of :func:`thing_to_id` for translating multiple things
        at once.

        :param things: Multiple things to be translated to IDs. Must be already
         in the table's set of things.
        :returns: A numpy array of the same integers that :func:`thing_to_id` would
         return.
        """
        ret = np.empty(len(things), dtype=np.int32)
        for i in range(len(things)):
            ret[i] = self.thing_to_id(things[i])
        return ret
