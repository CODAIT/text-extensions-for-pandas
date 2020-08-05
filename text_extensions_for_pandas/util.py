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

#
# util.py
#
# Part of text_extensions_for_pandas
#
# Internal utility functions, not exposed in the public API.
#

import numpy as np
from typing import *
import unittest

# Internal imports

_ELLIPSIS = " [...] "
_ELLIPSIS_LEN = len(_ELLIPSIS)


class TestBase(unittest.TestCase):
    """
    Base class to hold common utility code used by test cases in multiple files.
    """

    def _assertArrayEquals(self, a1: Union[np.ndarray, List[Any]],
                           a2: Union[np.ndarray, List[Any]]) -> None:
        """
        Assert that two arrays are completely identical, with useful error
        messages if they are not.

        :param a1: first array to compare. Lists automatically converted to
         arrays.
        :param a2: second array (or list)
        """
        a1 = np.array(a1) if isinstance(a1, np.ndarray) else a1
        a2 = np.array(a2) if isinstance(a2, np.ndarray) else a2
        if len(a1) != len(a2):
            raise self.failureException(
                f"Arrays:\n"
                f"   {a1}\n"
                f"and\n"
                f"   {a2}\n"
                f"have different lengths {len(a1)} and {len(a2)}"
            )
        mask = (a1 == a2)
        if not np.all(mask):
            raise self.failureException(
                f"Arrays:\n"
                f"   {a1}\n"
                f"and\n"
                f"   {a2}\n"
                f"differ at positions: {np.argwhere(~mask)}"
            )
