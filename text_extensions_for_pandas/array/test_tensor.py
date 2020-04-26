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

import textwrap
import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from text_extensions_for_pandas.array.tensor import TensorArray


class TestTensor(unittest.TestCase):

    def setUp(self):
        # Ensure that diffs are consistent
        pd.set_option("display.max_columns", 250)

    def tearDown(self):
        pd.reset_option("display.max_columns")

    def test_create(self):
        x = np.ones([5, 2, 3])
        s = TensorArray(x)
        self.assertEqual(len(s), 5)

        x = [np.ones([2, 3])] * 5
        s = TensorArray(x)
        self.assertEqual(len(s), 5)

        x = [np.ones([2, 3]), np.ones([3, 2])]
        with self.assertRaises(ValueError):
            TensorArray(x)

        with self.assertRaises(TypeError):
            TensorArray(2112)

    def test_operations(self):
        x = np.ones([5, 3])

        # equal
        s1 = TensorArray(x)
        s2 = TensorArray(x)
        self.assertTrue(np.all(s1 == s2))

        # less, greater
        s2 = TensorArray(x * 2)
        self.assertTrue(np.all(s1 < s2))
        self.assertFalse(np.any(s1 > s2))

        # add TensorArrays
        s1 = TensorArray(x * 2)
        s2 = TensorArray(x * 3)
        result = s1 + s2
        self.assertTrue(isinstance(result, TensorArray))
        npt.assert_equal(result.to_numpy().shape, [5, 3])
        self.assertTrue(np.all(result == 5))

        # multiply TensorArrays
        s1 = TensorArray(x * 2)
        s2 = TensorArray(x * 3)
        result = s1 * s2
        npt.assert_equal(result.to_numpy().shape, [5, 3])
        self.assertTrue(np.all(result == 6))

        # multiply scalar
        s1 = TensorArray(x * 2)
        result = s1 * 4
        npt.assert_equal(result.to_numpy().shape, [5, 3])
        self.assertTrue(np.all(result == 8))

    def test_setitem(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        a = TensorArray(x)
        a[1] = np.array([42, 42])
        npt.assert_equal(a[1], [42, 42])

    def test_repr(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected = textwrap.dedent(
            """\
        array([[1, 2],
               [3, 4],
               [5, 6]])"""
        )
        s = TensorArray(x)
        result = s.__repr__()
        self.assertEqual(expected, result)

        result = repr(pd.Series(s))
        expected = textwrap.dedent(
            """\
            0   [1 2]
            1   [3 4]
            2   [5 6]
            dtype: TensorType"""
        )
        self.assertEqual(expected, result)

        # The following currently doesn't work, due to
        # https://github.com/pandas-dev/pandas/issues/33770
        # TODO: Re-enable when a version of Pandas with a fix is released.
        # y = np.array([[True, False], [False, True], [False, False]])
        # s = TensorArray(y)
        # result = s.__repr__()
        # expected = textwrap.dedent(
        #     """\
        #     array([[ True, False],
        #            [False,  True],
        #            [False, False]])"""
        # )
        # self.assertEqual(expected, result)
        #
        # series = pd.Series(s)
        # result = repr(series)
        # expected = textwrap.dedent(
        #     """\
        #     0   [ True False]
        #     1   [False  True]
        #     2   [False False]
        #     dtype: TensorType"""
        # )
        # # self.assertEqual(expected, result)
        # print(f"***{result}***")

    def test_to_str(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected = '[[1 2]\n [3 4]\n [5 6]]'
        s = TensorArray(x)
        result = str(s)
        self.assertEqual(expected, result)

        # The following currently doesn't work, due to
        # https://github.com/pandas-dev/pandas/issues/33770
        # TODO: Re-enable when a version of Pandas with a fix is released.
        # y = np.array([[True, False], [False, True], [False, False]])
        # s = TensorArray(y)
        # result = str(s)
        # expected = textwrap.dedent(
        #     """\
        #     [[ True False]
        #      [False  True]
        #      [False False]]"""
        # )
        # self.assertEqual(expected, result)

    def test_concat(self):
        x = np.arange(6).reshape((3, 2))
        y = np.arange(6, 12).reshape((3, 2))
        x_arr = TensorArray(x)
        y_arr = TensorArray(y)
        concat_arr = TensorArray._concat_same_type((x_arr, y_arr))
        result = str(concat_arr)
        self.assertEqual(
            result,
            textwrap.dedent(
                """\
                [[ 0  1]
                 [ 2  3]
                 [ 4  5]
                 [ 6  7]
                 [ 8  9]
                 [10 11]]""")
        )

    def test_series_to_str(self):
        x = np.arange(50).reshape((10, 5))
        a = TensorArray(x)
        s = pd.Series(a)
        result = s.to_string(max_rows=4)
        self.assertEqual(
            result,
            textwrap.dedent(
                                """\
                0        [0 1 2 3 4]
                1        [5 6 7 8 9]
                          ...       
                8   [40 41 42 43 44]
                9   [45 46 47 48 49]""")
        )

    def test_slice(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)

        result = s[1]
        expected = np.array([3, 4])
        npt.assert_array_equal(expected, result)

        result = s[1:3]
        expected = np.array([[3, 4], [5, 6]])
        npt.assert_array_equal(expected, result)

    def test_asarray(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)
        a = np.asarray(s)
        npt.assert_array_equal(x, a)
        npt.assert_array_equal(x, s.to_numpy())


class TensorArrayDataFrameTests(unittest.TestCase):

    def test_create(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)
        df = pd.DataFrame({'i': list(range(len(x))), 'tensor': s})
        # TODO
        print(df)

    def test_sum(self):
        keys = ["a", "a", "b", "c", "c", "c"]
        values = np.array([[1, 1]] * len(keys))
        df = pd.DataFrame({
            "key": keys,
            "value": TensorArray(values)
        })
        result_df = df.groupby("key").aggregate({"value": "sum"})
        self.assertEqual(
            repr(result_df),
            textwrap.dedent(
                """\
                      value
                key        
                a    [2, 2]
                b    [1, 1]
                c    [3, 3]""")
        )

        # 2D values
        values2 = np.array([[[1, 1], [1, 1]]] * len(keys))
        df2 = pd.DataFrame({
            "key": keys,
            "value": TensorArray(values2)
        })
        result2_df = df2.groupby("key").aggregate({"value": "sum"})
        self.assertEqual(
            repr(result2_df),
            textwrap.dedent(
                """\
                                value
                key                  
                a    [[2, 2], [2, 2]]
                b    [[1, 1], [1, 1]]
                c    [[3, 3], [3, 3]]""")
        )