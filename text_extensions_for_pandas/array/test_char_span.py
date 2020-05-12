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
import os
import tempfile
import unittest

from text_extensions_for_pandas.array.char_span import *
from text_extensions_for_pandas.util import TestBase


class CharSpanTest(unittest.TestCase):
    def test_create(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 0, 4)
        self.assertEqual(s1.covered_text, "This")

        # Begin too small
        with self.assertRaises(ValueError):
            CharSpan(test_text, -2, 4)

        # End too small
        with self.assertRaises(ValueError):
            CharSpan(test_text, 1, -1)

        # Begin null, end not null
        with self.assertRaises(ValueError):
            CharSpan(test_text, CharSpan.NULL_OFFSET_VALUE, 0)

    def test_repr(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 0, 4)
        self.assertEqual(repr(s1), "[0, 4): 'This'")

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

    def test_overlaps(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 2, 4)
        no_overlap = [
            CharSpan(test_text, 0, 1),
            CharSpan(test_text, 0, 2),
            CharSpan(test_text, 1, 1),
            CharSpan(test_text, 2, 2),
            CharSpan(test_text, 4, 4),
            CharSpan(test_text, 4, 5),
            CharSpan(test_text, 5, 7),
        ]
        overlap = [
            CharSpan(test_text, 1, 3),
            CharSpan(test_text, 1, 4),
            CharSpan(test_text, 2, 3),
            CharSpan(test_text, 2, 4),
            CharSpan(test_text, 2, 5),
            CharSpan(test_text, 3, 3),
            CharSpan(test_text, 3, 4),
        ]

        for s_other in no_overlap:
            self.assertFalse(s1.overlaps(s_other))
            self.assertFalse(s_other.overlaps(s1))

        for s_other in overlap:
            self.assertTrue(s1.overlaps(s_other))
            self.assertTrue(s_other.overlaps(s1))

        s2 = CharSpan(test_text, 1, 1)
        s3 = CharSpan(test_text, 1, 1)
        s4 = CharSpan(test_text, 2, 2)
        self.assertTrue(s2.overlaps(s3))
        self.assertFalse(s3.overlaps(s4))

    def test_contains(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 2, 4)
        not_contained = [
            CharSpan(test_text, 0, 1),
            CharSpan(test_text, 0, 2),
            CharSpan(test_text, 1, 1),
            CharSpan(test_text, 1, 3),
            CharSpan(test_text, 1, 4),
            CharSpan(test_text, 2, 5),
            CharSpan(test_text, 4, 5),
            CharSpan(test_text, 5, 7),
        ]
        contained = [
            CharSpan(test_text, 2, 2),
            CharSpan(test_text, 2, 3),
            CharSpan(test_text, 2, 4),
            CharSpan(test_text, 3, 3),
            CharSpan(test_text, 3, 4),
            CharSpan(test_text, 4, 4),
        ]

        for s_other in not_contained:
            self.assertFalse(s1.contains(s_other))

        for s_other in contained:
            self.assertTrue(s1.contains(s_other))

        s2 = CharSpan(test_text, 1, 1)
        s3 = CharSpan(test_text, 1, 1)
        s4 = CharSpan(test_text, 2, 2)
        self.assertTrue(s2.contains(s3))
        self.assertFalse(s3.contains(s4))

    def test_context(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 5, 7)
        s2 = CharSpan(test_text, 0, 4)
        s3 = CharSpan(test_text, 10, 15)

        self.assertEqual(s1.context(), "This [is] a test.")
        self.assertEqual(s1.context(2), "...s [is] a...")
        self.assertEqual(s2.context(), "[This] is a test.")
        self.assertEqual(s2.context(3), "[This] is...")
        self.assertEqual(s3.context(), "This is a [test.]")
        self.assertEqual(s3.context(3), "... a [test.]")


class ArrayTestBase(TestBase):
    """
    Shared base class for CharSpanArrayTest and TokenSpanArrayTest
    """

    @staticmethod
    def _make_spans_of_tokens():
        """
        :return: An example CharSpanArray containing the tokens of the string
          "This is a test.", not including the period at the end.
        """
        return CharSpanArray(
            "This is a test.", np.array([0, 5, 8, 10]), np.array([4, 7, 9, 14])
        )


class CharSpanArrayTest(ArrayTestBase):
    def test_create(self):
        arr = self._make_spans_of_tokens()
        self._assertArrayEquals(arr.covered_text, ["This", "is", "a", "test"])

    def test_dtype(self):
        arr = CharSpanArray("", np.array([0],), np.array([0]))
        self.assertTrue(isinstance(arr.dtype, CharSpanType))

    def test_len(self):
        self.assertEqual(len(self._make_spans_of_tokens()), 4)

    def test_getitem(self):
        arr = self._make_spans_of_tokens()
        self.assertEqual(arr[2].covered_text, "a")
        self._assertArrayEquals(arr[2:4].covered_text, ["a", "test"])

    def test_setitem(self):
        arr = self._make_spans_of_tokens()
        arr[1] = arr[2]
        self._assertArrayEquals(arr.covered_text, ["This", "a", "a", "test"])
        arr[3] = None
        self._assertArrayEquals(arr.covered_text, ["This", "a", "a", None])
        with self.assertRaises(ValueError):
            arr[0] = "Invalid argument for __setitem__()"

    def test_equals(self):
        arr = self._make_spans_of_tokens()
        self._assertArrayEquals(arr[0:4] == arr[1], [False, True, False, False])
        arr2 = self._make_spans_of_tokens()
        self._assertArrayEquals(arr == arr2, [True] * 4)

        self.assertTrue(arr.equals(arr2))
        arr2._text = "This is a different string."
        arr2.increment_version()
        self.assertFalse(arr.equals(arr2))
        arr2._text = arr.target_text
        arr2.increment_version()
        self.assertTrue(arr.equals(arr2))
        self.assertTrue(arr2.equals(arr))
        arr[2] = arr[1]
        self.assertFalse(arr.equals(arr2))
        self.assertFalse(arr2.equals(arr))
        arr[2] = arr2[2]
        self.assertTrue(arr.equals(arr2))
        self.assertTrue(arr2.equals(arr))

    def test_not_equals(self):
        arr = self._make_spans_of_tokens()
        self._assertArrayEquals(arr[0:4] != arr[1], [True, False, True, True])
        arr2 = self._make_spans_of_tokens()
        self._assertArrayEquals(arr != arr2, [False] * 4)

    def test_nulls(self):
        arr = self._make_spans_of_tokens()
        arr[2] = CharSpan(
            arr.target_text, CharSpan.NULL_OFFSET_VALUE, CharSpan.NULL_OFFSET_VALUE
        )
        self.assertIsNone(arr.covered_text[2])
        self._assertArrayEquals(arr.covered_text, ["This", "is", None, "test"])
        self._assertArrayEquals(arr.isna(), [False, False, True, False])

    def test_copy(self):
        arr = self._make_spans_of_tokens()
        arr2 = arr.copy()
        arr[0] = CharSpan(arr.target_text, 8, 9)
        self._assertArrayEquals(arr2.covered_text, ["This", "is", "a", "test"])
        self._assertArrayEquals(arr.covered_text, ["a", "is", "a", "test"])

    def test_take(self):
        arr = self._make_spans_of_tokens()
        self._assertArrayEquals(arr.take([2, 0]).covered_text, ["a", "This"])
        self._assertArrayEquals(arr.take([]).covered_text, [])
        self._assertArrayEquals(
            arr.take([2, -1, 0]).covered_text, ["a", "test", "This"]
        )
        self._assertArrayEquals(
            arr.take([2, -1, 0], allow_fill=True).covered_text, ["a", None, "This"]
        )

    def test_less_than(self):
        arr1 = CharSpanArray(
            "This is a test.", np.array([0, 5, 8, 10]), np.array([4, 7, 9, 14])
        )
        s1 = CharSpan(arr1.target_text, 0, 1)
        s2 = CharSpan(arr1.target_text, 11, 14)
        arr2 = CharSpanArray(arr1.target_text, [0, 3, 10, 7], [0, 4, 12, 9])

        self._assertArrayEquals(s1 < arr1, [False, True, True, True])
        self._assertArrayEquals(s2 > arr1, [True, True, True, False])
        self._assertArrayEquals(arr1 < s1, [False, False, False, False])
        self._assertArrayEquals(arr1 < arr2, [False, False, True, False])

    def test_reduce(self):
        arr = self._make_spans_of_tokens()
        self.assertEqual(arr._reduce("sum"), CharSpan(arr.target_text, 0, 14))
        # Remind ourselves to modify this test after implementing min and max
        with self.assertRaises(TypeError):
            arr._reduce("min")

    def test_as_tuples(self):
        arr = CharSpanArray(
            "This is a test.", np.array([0, 5, 8, 10]), np.array([4, 7, 9, 14])
        )
        self._assertArrayEquals(arr.as_tuples(), [[0, 4], [5, 7], [8, 9], [10, 14]])

    def test_normalized_covered_text(self):
        arr = self._make_spans_of_tokens()
        self._assertArrayEquals(
            arr.normalized_covered_text, ["this", "is", "a", "test"]
        )

    def test_as_frame(self):
        arr = self._make_spans_of_tokens()
        df = arr.as_frame()
        self._assertArrayEquals(df.columns, ["begin", "end", "covered_text"])
        self.assertEqual(len(df), len(arr))

    def test_overlaps(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 2, 4)
        s_others = [
            # Non-overlapping
            CharSpan(test_text, 0, 1),
            CharSpan(test_text, 0, 2),
            CharSpan(test_text, 1, 1),
            CharSpan(test_text, 2, 2),
            CharSpan(test_text, 4, 4),
            CharSpan(test_text, 4, 5),
            CharSpan(test_text, 5, 7),
            # Overlapping
            CharSpan(test_text, 1, 3),
            CharSpan(test_text, 1, 4),
            CharSpan(test_text, 2, 3),
            CharSpan(test_text, 2, 4),
            CharSpan(test_text, 2, 5),
            CharSpan(test_text, 3, 3),
            CharSpan(test_text, 3, 4),
        ]

        expected_mask = [False] * 7 + [True] * 7

        s1_array = CharSpanArray._from_sequence([s1] * len(s_others))
        others_array = CharSpanArray._from_sequence(s_others)

        self._assertArrayEquals(s1_array.overlaps(others_array), expected_mask)
        self._assertArrayEquals(others_array.overlaps(s1_array), expected_mask)
        self._assertArrayEquals(others_array.overlaps(s1), expected_mask)

        # Check zero-length span cases
        one_one = CharSpanArray._from_sequence([CharSpan(test_text, 1, 1)] * 2)
        one_one_2_2 = CharSpanArray._from_sequence(
            [CharSpan(test_text, 1, 1), CharSpan(test_text, 2, 2)]
        )
        self._assertArrayEquals(one_one.overlaps(one_one_2_2), [True, False])
        self._assertArrayEquals(one_one_2_2.overlaps(one_one), [True, False])
        self._assertArrayEquals(
            one_one_2_2.overlaps(CharSpan(test_text, 1, 1)), [True, False]
        )

    def test_contains(self):
        test_text = "This is a test."
        s1 = CharSpan(test_text, 2, 4)
        s_others = [
            # Not contained within s1
            CharSpan(test_text, 0, 1),
            CharSpan(test_text, 0, 2),
            CharSpan(test_text, 1, 1),
            CharSpan(test_text, 1, 3),
            CharSpan(test_text, 1, 4),
            CharSpan(test_text, 2, 5),
            CharSpan(test_text, 4, 5),
            CharSpan(test_text, 5, 7),
            # Contained within s1
            CharSpan(test_text, 2, 2),
            CharSpan(test_text, 2, 3),
            CharSpan(test_text, 2, 4),
            CharSpan(test_text, 3, 3),
            CharSpan(test_text, 3, 4),
            CharSpan(test_text, 4, 4),
        ]

        expected_mask = [False] * 8 + [True] * 6

        s1_array = CharSpanArray._from_sequence([s1] * len(s_others))
        others_array = CharSpanArray._from_sequence(s_others)

        self._assertArrayEquals(s1_array.contains(others_array), expected_mask)

        # Check zero-length span cases
        one_one = CharSpanArray._from_sequence([CharSpan(test_text, 1, 1)] * 2)
        one_one_2_2 = CharSpanArray._from_sequence(
            [CharSpan(test_text, 1, 1), CharSpan(test_text, 2, 2)]
        )
        self._assertArrayEquals(one_one.contains(one_one_2_2), [True, False])
        self._assertArrayEquals(one_one_2_2.contains(one_one), [True, False])
        self._assertArrayEquals(
            one_one_2_2.contains(CharSpan(test_text, 1, 1)), [True, False]
        )


class CharSpanArrayIOTests(ArrayTestBase):

    def test_feather(self):
        arr = self._make_spans_of_tokens()
        df = pd.DataFrame({'CharSpan': arr})

        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, 'char_span_array_test.feather')
            df.to_feather(filename)
            df_read = pd.read_feather(filename)
            pd.testing.assert_frame_equal(df, df_read)


if __name__ == "__main__":
    unittest.main()
