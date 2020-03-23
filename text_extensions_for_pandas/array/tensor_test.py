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

import numpy as np
import pandas as pd
import unittest

from text_extensions_for_pandas.array.tensor import *


class TensorTest(unittest.TestCase):
    def test_create(self):
        x = np.random.rand(10, 3)
        a = TensorArray(x)

        #self.assertEqual(s1.covered_text, "This")

        b = a[1]
        c = a[2:4]

        df = pd.DataFrame({'i': list(range(10)), 'a': a})
        print(df)

    '''
    def test_equals(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 0, 4)
        s2 = CharSpan(test_text, 0, 4)
        s3 = CharSpan(test_text, 0, 5)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_less_than(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 0, 4)
        s2 = CharSpan(test_text, 2, 4)
        s3 = CharSpan(test_text, 4, 5)

        self.assertLess(s1, s3)
        self.assertLessEqual(s1, s3)
        self.assertFalse(s1 < s2)
    '''
'''
class ArrayTestBase(unittest.TestCase):
    """
    Shared base class for CharSpanArrayTest and TokenSpanArrayTest
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
        mask = (a1 == a2)
        if not np.all(mask):
            raise self.failureException(
                f"Arrays:\n"
                f"   {a1}\n"
                f"and\n"
                f"   {a2}\n"
                f"differ at positions: {np.argwhere(mask)}"
            )

    @staticmethod
    def _make_char_spans():
        """
        :return: An example CharSpanArray containing the tokens of the string
          "This is a test.", not including the period at the end.
        """
        return CharSpanArray("This is a test.",
                             np.array([0, 5, 8, 10]),
                             np.array([4, 7, 9, 14]))


class CharSpanArrayTest(ArrayTestBase):

    def test_create(self):
        arr = self._make_char_spans()
        self._assertArrayEquals(
            arr.covered_text, ["This", "is", "a", "test"])

    def test_nulls(self):
        arr = self._make_char_spans()
        arr[2] = CharSpan(arr.target_text, CharSpan.NULL_OFFSET_VALUE,
                          CharSpan.NULL_OFFSET_VALUE)
        self.assertIsNone(arr.covered_text[2])
        self._assertArrayEquals(
            arr.covered_text, ["This", "is", None, "test"])
        self._assertArrayEquals(
            arr.isna(), [False, False, True, False])

    def test_copy(self):
        arr = self._make_char_spans()
        arr2 = arr.copy()
        arr[0] = CharSpan(arr.target_text, 8, 9)
        self._assertArrayEquals(arr2.covered_text,
                                ["This", "is", "a", "test"])
        self._assertArrayEquals(arr.covered_text,
                                ["a", "is", "a", "test"])

    def test_take(self):
        arr = self._make_char_spans()
        self._assertArrayEquals(arr.take([2, 0]).covered_text,
                                ["a", "This"])
        self._assertArrayEquals(arr.take([]).covered_text, [])
        self._assertArrayEquals(arr.take([2, -1, 0]).covered_text,
                                ["a", "test", "This"])
        self._assertArrayEquals(
            arr.take([2, -1, 0], allow_fill=True).covered_text,
            ["a", None, "This"])

    def test_dtype(self):
        arr = CharSpanArray("", np.array([]), np.array([]))
        self.assertTrue(isinstance(arr.dtype, CharSpanType))

    def test_len(self):
        self.assertEqual(len(self._make_char_spans()), 4)

    def test_less_than(self):
        arr1 = CharSpanArray("This is a test.",
                             np.array([0, 5, 8, 10]),
                             np.array([4, 7, 9, 14]))
        s1 = CharSpan(arr1.target_text, 0, 1)
        s2 = CharSpan(arr1.target_text, 11, 14)
        arr2 = CharSpanArray(arr1.target_text,
                             [0, 3, 10, 7],
                             [0, 4, 12, 9])

        self._assertArrayEquals(s1 < arr1, [False, True, True, True])
        self._assertArrayEquals(s2 > arr1, [True, True, True, False])
        self._assertArrayEquals(arr1 < s1, [False, False, False, False])
        self._assertArrayEquals(arr1 < arr2, [False, False, True, False])

    def test_getitem(self):
        arr = self._make_char_spans()
        self.assertEqual(arr[2].covered_text, "a")
        self._assertArrayEquals(arr[2:4].covered_text, ["a", "test"])

    def test_setitem(self):
        arr = self._make_char_spans()
        arr[1] = arr[2]
        self._assertArrayEquals(arr.covered_text,
                                ["This", "a", "a", "test"])
        arr[3] = None
        self._assertArrayEquals(arr.covered_text,
                                ["This", "a", "a", None])
        with self.assertRaises(ValueError):
            arr[0] = "Invalid argument for __setitem__()"

    def test_reduce(self):
        arr = self._make_char_spans()
        self.assertEqual(
            arr._reduce("sum"), CharSpan(arr.target_text, 0, 14))
        # Remind ourselves to modify this test after implementing min and max
        with self.assertRaises(TypeError):
            arr._reduce("min")

    def test_as_tuples(self):
        arr = CharSpanArray("This is a test.",
                            np.array([0, 5, 8, 10]),
                            np.array([4, 7, 9, 14]))
        self._assertArrayEquals(
            arr.as_tuples(),
            [[0, 4], [5, 7], [8, 9], [10, 14]]
        )

    def test_normalized_covered_text(self):
        arr = self._make_char_spans()
        print(f"Covered text: {arr.covered_text}")
        print(f"Normalized covered text: {arr.normalized_covered_text}")
        self._assertArrayEquals(
            arr.normalized_covered_text, ["this", "is", "a", "test"])

    def test_as_frame(self):
        arr = self._make_char_spans()
        df = arr.as_frame()
        self._assertArrayEquals(df.columns, ["begin", "end", "covered_text"])
        self.assertEqual(len(df), len(arr))
'''