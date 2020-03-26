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


class TensorTest(unittest.TestCase):

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

        s1 = TensorArray(x)
        s2 = TensorArray(x)
        self.assertTrue(np.all(s1 == s2))

        s2 = TensorArray(x * 2)
        self.assertTrue(np.all(s1 < s2))
        self.assertFalse(np.any(s1 > s2))

    def test_repr(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected = """\
array([[1, 2],
       [3, 4],
       [5, 6]])"""
        expected = textwrap.dedent(
            """\
        array([[1, 2],
               [3, 4],
               [5, 6]])"""
        )
        s = TensorArray(x)
        result = s.__repr__()
        self.assertEqual(expected, result)

    def test_to_str(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected = '[[1 2]\n [3 4]\n [5 6]]'
        s = TensorArray(x)
        result = str(s)
        self.assertEqual(expected, result)

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
