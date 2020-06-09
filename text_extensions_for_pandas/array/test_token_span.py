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

from text_extensions_for_pandas.array.test_char_span import ArrayTestBase

from text_extensions_for_pandas.array.token_span import *


class TokenSpanTest(ArrayTestBase):
    def test_create(self):
        toks = self._make_spans_of_tokens()
        s1 = TokenSpan(toks, 0, 1)
        self.assertEqual(s1.covered_text, "This")

        # Begin too small
        with self.assertRaises(ValueError):
            TokenSpan(toks, -2, 4)

        # End too small
        with self.assertRaises(ValueError):
            TokenSpan(toks, 1, -1)

        # End too big
        with self.assertRaises(ValueError):
            TokenSpan(toks, 1, 10)

        # Begin null, end not null
        with self.assertRaises(ValueError):
            TokenSpan(toks, TokenSpan.NULL_OFFSET_VALUE, 0)

    def test_repr(self):
        toks = self._make_spans_of_tokens()
        s1 = TokenSpan(toks, 0, 2)
        self.assertEqual(repr(s1), "[0, 7): 'This is'")

        toks2 = CharSpanArray(
            "This is a really really really really really really really really "
            "really long string.",
            np.array([0, 5, 8, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 78, 84]),
            np.array([4, 7, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 77, 84, 85]),
        )
        self._assertArrayEquals(
            toks2.covered_text,
            [
                "This",
                "is",
                "a",
                "really",
                "really",
                "really",
                "really",
                "really",
                "really",
                "really",
                "really",
                "really",
                "long",
                "string",
                ".",
            ],
        )
        s2 = TokenSpan(toks2, 0, 4)
        self.assertEqual(repr(s2), "[0, 16): 'This is a really'")
        s2 = TokenSpan(toks2, 0, 15)
        self.assertEqual(
            repr(s2),
            "[0, 85): 'This is a really really really really really really "
            "really really really [...]'"
        )

    def test_equals(self):
        toks = self._make_spans_of_tokens()
        other_toks = toks[:-1].copy()
        s1 = TokenSpan(toks, 0, 2)
        s2 = TokenSpan(toks, 0, 2)
        s3 = TokenSpan(toks, 0, 3)
        s4 = TokenSpan(other_toks, 0, 2)
        s5 = CharSpan(toks.target_text, s4.begin, s4.end)
        s6 = CharSpan(toks.target_text, s4.begin, s4.end + 1)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertEqual(s1, s4)
        self.assertEqual(s1, s5)
        self.assertEqual(s5, s1)
        self.assertNotEqual(s1, s6)

    def test_less_than(self):
        toks = self._make_spans_of_tokens()
        s1 = TokenSpan(toks, 0, 3)
        s2 = TokenSpan(toks, 2, 3)
        s3 = TokenSpan(toks, 3, 4)

        self.assertLess(s1, s3)
        self.assertLessEqual(s1, s3)
        self.assertFalse(s1 < s2)

    def test_add(self):
        toks = self._make_spans_of_tokens()
        s1 = TokenSpan(toks, 0, 3)
        s2 = TokenSpan(toks, 2, 3)
        s3 = TokenSpan(toks, 3, 4)

        self.assertEqual(s1 + s2, s1)
        self.assertEqual(s2 + s3, TokenSpan(toks, 2, 4))

    def test_hash(self):
        toks = self._make_spans_of_tokens()
        s1 = TokenSpan(toks, 0, 3)
        s2 = TokenSpan(toks, 0, 3)
        s3 = TokenSpan(toks, 3, 4)
        d = {s1: "foo"}
        self.assertEqual(d[s1], "foo")
        self.assertEqual(d[s2], "foo")
        d[s2] = "bar"
        d[s3] = "fab"
        self.assertEqual(d[s1], "bar")
        self.assertEqual(d[s2], "bar")
        self.assertEqual(d[s3], "fab")


class TokenSpanArrayTest(ArrayTestBase):
    def _make_spans(self):
        toks = self._make_spans_of_tokens()
        return TokenSpanArray(toks, [0, 1, 2, 3, 0, 2, 0], [1, 2, 3, 4, 2, 4, 4])

    def test_create(self):
        arr = self._make_spans()
        self._assertArrayEquals(
            arr.covered_text,
            ["This", "is", "a", "test", "This is", "a test", "This is a test"],
        )

        with self.assertRaises(TypeError):
            TokenSpanArray(self._make_spans_of_tokens(), "Not a valid begins list", [42])

    def test_dtype(self):
        arr = self._make_spans()
        self.assertTrue(isinstance(arr.dtype, TokenSpanType))

    def test_len(self):
        self.assertEqual(len(self._make_spans()), 7)

    def test_getitem(self):
        arr = self._make_spans()
        self.assertEqual(arr[2].covered_text, "a")
        self._assertArrayEquals(arr[2:4].covered_text, ["a", "test"])

    def test_setitem(self):
        arr = self._make_spans()
        arr[1] = arr[2]
        self._assertArrayEquals(arr.covered_text[0:4], ["This", "a", "a", "test"])
        arr[3] = None
        self._assertArrayEquals(arr.covered_text[0:4], ["This", "a", "a", None])
        with self.assertRaises(ValueError):
            arr[0] = "Invalid argument for __setitem__()"

        arr[0:2] = arr[0]
        self._assertArrayEquals(arr.covered_text[0:4], ["This", "This", "a", None])
        arr[[0, 1, 3]] = None
        self._assertArrayEquals(arr.covered_text[0:4], [None, None, "a", None])
        arr[[2, 1, 3]] = arr[[4, 5, 6]]
        self._assertArrayEquals(
            arr.covered_text[0:4], [None, "a test", "This is", "This is a test"]
        )

    def test_equals(self):
        arr = self._make_spans()
        self._assertArrayEquals(arr[0:4] == arr[1], [False, True, False, False])
        arr2 = self._make_spans()
        self._assertArrayEquals(arr == arr, [True] * 7)
        self._assertArrayEquals(arr == arr2, [True] * 7)
        self._assertArrayEquals(arr[0:3] == arr[3:6], [False, False, False])
        arr3 = CharSpanArray(arr.target_text, arr.begin, arr.end)
        self._assertArrayEquals(arr == arr3, [True] * 7)
        self._assertArrayEquals(arr3 == arr, [True] * 7)

    def test_not_equals(self):
        arr = self._make_spans()
        arr2 = self._make_spans()
        self._assertArrayEquals(arr[0:4] != arr[1], [True, False, True, True])
        self._assertArrayEquals(arr != arr2, [False] * 7)
        self._assertArrayEquals(arr[0:3] != arr[3:6], [True, True, True])

    def test_concat_same_type(self):
        arr = self._make_spans()
        arr2 = self._make_spans()
        # Type: TokenSpanArray
        arr3 = TokenSpanArray._concat_same_type((arr, arr2))
        self._assertArrayEquals(arr3.covered_text, np.tile(arr2.covered_text, 2))

    def test_from_factorized(self):
        arr = self._make_spans()
        spans_list = [arr[i] for i in range(len(arr))]
        arr2 = TokenSpanArray._from_factorized(spans_list, arr)
        self._assertArrayEquals(arr.covered_text, arr2.covered_text)

    def test_from_sequence(self):
        arr = self._make_spans()
        spans_list = [arr[i] for i in range(len(arr))]
        arr2 = TokenSpanArray._from_sequence(spans_list)
        self._assertArrayEquals(arr.covered_text, arr2.covered_text)

    def test_nulls(self):
        arr = self._make_spans()
        self._assertArrayEquals(arr.isna(), [False] * 7)
        self.assertFalse(arr.have_nulls)
        arr[2] = TokenSpan.make_null(arr.tokens)
        self.assertIsNone(arr.covered_text[2])
        self._assertArrayEquals(arr[0:4].covered_text, ["This", "is", None, "test"])
        self._assertArrayEquals(arr[0:4].isna(), [False, False, True, False])
        self.assertTrue(arr.have_nulls)

    def test_copy(self):
        arr = self._make_spans()
        arr2 = arr.copy()
        self._assertArrayEquals(arr.covered_text, arr2.covered_text)
        self.assertEqual(arr[1], arr2[1])
        arr[1] = TokenSpan.make_null(arr.tokens)
        self.assertNotEqual(arr[1], arr2[1])

    # Double underscore because you can't call a test case "test_take"
    def test_take(self):
        arr = self._make_spans()
        arr2 = arr.take([1, 1, 2, 3, 5, -1])
        self._assertArrayEquals(
            arr2.covered_text, ["is", "is", "a", "test", "a test", "This is a test"]
        )
        arr3 = arr.take([1, 1, 2, 3, 5, -1], allow_fill=True)
        self._assertArrayEquals(
            arr3.covered_text, ["is", "is", "a", "test", "a test", None]
        )

    def test_less_than(self):
        tokens = self._make_spans_of_tokens()
        arr1 = TokenSpanArray(tokens, [0, 2], [4, 3])
        s1 = TokenSpan(tokens, 0, 1)
        s2 = TokenSpan(tokens, 3, 4)
        arr2 = TokenSpanArray(tokens, [0, 3], [0, 4])

        self._assertArrayEquals(s1 < arr1, [False, True])
        self._assertArrayEquals(s2 > arr1, [False, True])
        self._assertArrayEquals(arr1 < s1, [False, False])
        self._assertArrayEquals(arr1 < arr2, [False, True])

    def test_add(self):
        toks = self._make_spans_of_tokens()
        s1 = TokenSpan(toks, 0, 3)
        s2 = TokenSpan(toks, 2, 3)
        s3 = TokenSpan(toks, 3, 4)
        s4 = TokenSpan(toks, 2, 4)
        s5 = TokenSpan(toks, 0, 3)

        self._assertArrayEquals(
            TokenSpanArray._from_sequence([s1, s2, s3])
            + TokenSpanArray._from_sequence([s2, s3, s3]),
            TokenSpanArray._from_sequence([s1, s4, s3]),
        )
        self._assertArrayEquals(
            TokenSpanArray._from_sequence([s1, s2, s3]) + s2,
            TokenSpanArray._from_sequence([s5, s2, s4]),
        )

    def test_reduce(self):
        arr = self._make_spans()
        self.assertEqual(arr._reduce("sum"), TokenSpan(arr.tokens, 0, 4))
        # Remind ourselves to modify this test after implementing min and max
        with self.assertRaises(TypeError):
            arr._reduce("min")

    def test_make_array(self):
        arr = self._make_spans()
        arr_series = pd.Series(arr)
        toks_list = [arr[0], arr[1], arr[2], arr[3]]
        self._assertArrayEquals(
            TokenSpanArray.make_array(arr).covered_text,
            ["This", "is", "a", "test", "This is", "a test", "This is a test"],
        )
        self._assertArrayEquals(
            TokenSpanArray.make_array(arr_series).covered_text,
            ["This", "is", "a", "test", "This is", "a test", "This is a test"],
        )
        self._assertArrayEquals(
            TokenSpanArray.make_array(toks_list).covered_text,
            ["This", "is", "a", "test"],
        )

    def test_begin_and_end(self):
        arr = self._make_spans()
        self._assertArrayEquals(arr.begin, [0, 5, 8, 10, 0, 8, 0])
        self._assertArrayEquals(arr.end, [4, 7, 9, 14, 7, 14, 14])

    def test_normalized_covered_text(self):
        arr = self._make_spans()
        self._assertArrayEquals(
            arr.normalized_covered_text,
            ["this", "is", "a", "test", "this is", "a test", "this is a test"],
        )

    def test_as_frame(self):
        arr = self._make_spans()
        df = arr.as_frame()
        self._assertArrayEquals(
            df.columns, ["begin", "end", "begin_token", "end_token", "covered_text"]
        )
        self.assertEqual(len(df), len(arr))


class TokenSpanArrayIOTests(ArrayTestBase):

    def do_roundtrip(self, df):
        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, 'token_span_array_test.feather')
            df.to_feather(filename)
            df_read = pd.read_feather(filename)
            pd.testing.assert_frame_equal(df, df_read)

    def test_feather(self):
        toks = self._make_spans_of_tokens()

        # Equal token spans to tokens
        ts1 = TokenSpanArray(toks, np.arange(len(toks)), np.arange(len(toks)) + 1)
        df1 = pd.DataFrame({"ts1": ts1})
        self.do_roundtrip(df1)

        # More token spans than tokens
        ts2 = TokenSpanArray(toks, [0, 1, 2, 3, 0, 2, 0], [1, 2, 3, 4, 2, 4, 4])
        df2 = pd.DataFrame({"ts2": ts2})
        self.do_roundtrip(df2)

        # Less token spans than tokens, 2 splits no padding
        ts3 = TokenSpanArray(toks, [0, 3], [3, 4])
        df3 = pd.DataFrame({"ts3": ts3})
        self.do_roundtrip(df3)

        # Less token spans than tokens, 1 split with padding
        ts4 = TokenSpanArray(toks, [0, 2, 3], [2, 3, 4])
        df4 = pd.DataFrame({"ts4": ts4})
        self.do_roundtrip(df4)

        # With a CharSpan column, TokenSpan padded to same length
        df5 = pd.DataFrame({"cs": toks})
        df5 = pd.concat([df3, df5], axis=1)
        self.do_roundtrip(df5)

        # All columns together, TokenSpan arrays padded as needed
        df = pd.concat([df1, df2, df3, df4], axis=1)
        self.do_roundtrip(df)


if __name__ == "__main__":
    unittest.main()
