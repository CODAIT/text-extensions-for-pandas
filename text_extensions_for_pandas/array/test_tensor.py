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

import os
from distutils.version import LooseVersion
import tempfile
import textwrap
import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from pandas.tests.extension import base
import pyarrow as pa
import pytest

from text_extensions_for_pandas.array.tensor import TensorArray, TensorElement, TensorDtype


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

        x = np.empty((0, 2))
        s = TensorArray(x)
        self.assertEqual(len(s), 0)
        s = TensorArray([])
        self.assertEqual(len(s), 0)

        x = [np.ones([2, 3]), np.ones([3, 2])]
        with self.assertRaises(ValueError):
            TensorArray(x)

        # Copy constructor
        s_copy = s.copy()
        self.assertEqual(len(s), len(s_copy))

    def test_create_from_scalar(self):
        s = TensorArray(2112)
        self.assertEqual(len(s), 1)
        self.assertTupleEqual(s.numpy_shape, (1,))
        self.assertEqual(s[0], 2112)

        s = TensorArray(np.int64(2112))
        self.assertEqual(len(s), 1)
        self.assertTupleEqual(s.numpy_shape, (1,))
        self.assertEqual(s[0], 2112)

        x = np.array([1, 2112, 3])
        e = TensorElement(x[1])
        s = TensorArray(e)
        self.assertTupleEqual(s.numpy_shape, (1,))
        self.assertEqual(s[0], 2112)

    def test_create_from_scalar_list(self):
        x = [1, 2, 3, 4, 5]
        s = TensorArray(x)
        self.assertTupleEqual(s.numpy_shape, (len(x),))
        expected = np.array(x)
        npt.assert_array_equal(s.to_numpy(), expected)

        # Now with TensorElement values
        e = [TensorElement(np.array(i)) for i in x]
        s = pd.array(e, dtype=TensorDtype())
        npt.assert_array_equal(s.to_numpy(), expected)

        # Now with list of 1d tensors
        x = [np.array([i]) for i in x]
        s = pd.array(x, dtype=TensorDtype())
        self.assertTupleEqual(s.to_numpy().shape, (len(x), 1))
        npt.assert_array_equal(s.to_numpy(), np.array([[e] for e in expected]))

        # Pandas will create list of copies of the tensor element for the given indices
        s = pd.Series(np.nan, index=[0, 1, 2], dtype=TensorDtype())
        self.assertEqual(len(s), 3)
        self.assertTupleEqual(s.to_numpy().shape, (3,))
        result = s.isna()
        self.assertTrue(np.all(result.to_numpy()))

    def test_array_interface(self):
        # Extended version of Pandas TestPandasInterface.test_array_interface

        # Test scalar value
        s = TensorArray(3)
        result = np.array(s)
        self.assertTupleEqual(result.shape, (1,))
        expected = np.stack([np.asarray(i) for i in s])
        npt.assert_array_equal(result, expected)

        # Test scalar list
        s = TensorArray([1, 2, 3])
        result = np.array(s)
        self.assertTupleEqual(result.shape, (3,))
        expected = np.stack([np.asarray(i) for i in s])
        npt.assert_array_equal(result, expected)

        # Test 2d array
        s = TensorArray([[1], [2], [3]])
        result = np.array(s)
        self.assertTupleEqual(result.shape, (3, 1))
        expected = np.array([np.asarray(i) for i in s])
        npt.assert_array_equal(result, expected)

        # Test TensorElement
        x = [1, 2, 3]
        elements = [TensorElement(np.array(i)) for i in x]
        result = np.array([np.asarray(e) for e in elements])
        self.assertTupleEqual(result.shape, (3,))
        expected = np.array(x)
        npt.assert_array_equal(result, expected)

    def test_create_series(self):
        x = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]] * 100)
        a = TensorArray(x)
        s1 = pd.Series(a)
        s2 = pd.Series(a, dtype=TensorDtype())
        s3 = pd.Series(a, dtype=TensorDtype(), copy=True)
        self.assertEqual(len(x), len(s1))
        npt.assert_array_equal(x, s1.to_numpy())
        pdt.assert_series_equal(s1, s2)
        pdt.assert_series_equal(s1, s3)

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
        npt.assert_equal(result.numpy_shape, [5, 3])
        self.assertTrue(np.all(result == 5))

        # multiply TensorArrays
        s1 = TensorArray(x * 2)
        s2 = TensorArray(x * 3)
        result = s1 * s2
        npt.assert_equal(result.numpy_shape, [5, 3])
        self.assertTrue(np.all(result == 6))

        # multiply scalar
        s1 = TensorArray(x * 2)
        result = s1 * 4
        npt.assert_equal(result.numpy_shape, [5, 3])
        self.assertTrue(np.all(result == 8))

    def test_setitem(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        a = TensorArray(x)
        a[1] = np.array([42, 42])
        npt.assert_equal(a[1], [42, 42])

    def test_isna(self):
        expected = np.array([False, True, False, False])

        # Test numeric
        x = np.array([[1, 2], [np.nan, np.nan], [3, np.nan], [5, 6]])
        s = TensorArray(x)
        result = s.isna()
        npt.assert_equal(result, expected)

        # Test object
        d = {"a": 1}
        x = np.array([[d, d], None, [d, None], [d, d]], dtype=object)
        s = TensorArray(x)
        result = s.isna()
        npt.assert_equal(result, expected)

        # Test str
        x = np.array([["foo", "foo"], ["", ""], ["bar", ""], ["baz", "baz"]])
        s = TensorArray(x)
        result = s.isna()
        npt.assert_equal(result, expected)

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
            dtype: TensorDtype"""
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
        #     dtype: TensorDtype"""
        # )
        # # self.assertEqual(expected, result)
        # print(f"***{result}***")

    def test_to_str(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        expected = "[[1 2]\n [3 4]\n [5 6]]"
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
                 [10 11]]"""
            ),
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
                9   [45 46 47 48 49]"""
            ),
        )

    def test_slice(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)

        result = s[1]
        self.assertTrue(isinstance(result, TensorElement))
        expected = np.array([3, 4])
        npt.assert_array_equal(expected, result)

        result = s[1:3]
        self.assertTrue(isinstance(result, TensorArray))
        expected = np.array([[3, 4], [5, 6]])
        npt.assert_array_equal(expected, result.to_numpy())

    def test_bool_indexing(self):
        s = TensorArray([[1, 2], [3, 4]])

        result = s[[True, True]]
        self.assertTrue(isinstance(result, TensorArray))
        expected = np.array([[1, 2], [3, 4]])
        npt.assert_array_equal(result.to_numpy(), expected)

        result = s[[True, False]]
        self.assertTrue(isinstance(result, TensorArray))
        expected = np.array([[1, 2]])
        npt.assert_array_equal(result.to_numpy(), expected)

        result = s[[False, True]]
        self.assertTrue(isinstance(result, TensorArray))
        expected = np.array([[3, 4]])
        npt.assert_array_equal(result.to_numpy(), expected)

        result = s[[False, False]]
        self.assertTrue(isinstance(result, TensorArray))
        expected = np.empty((0, 2))
        npt.assert_array_equal(result.to_numpy(), expected)

    def test_asarray(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)
        a = np.asarray(s)
        npt.assert_array_equal(x, a)
        npt.assert_array_equal(x, s.to_numpy())

    def test_sum(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)
        df = pd.DataFrame({"s": s})

        sum_all = df["s"].sum()
        npt.assert_array_equal(sum_all.to_numpy(), [9, 12])

        sum_some = df["s"][[True, False, True]].sum()
        npt.assert_array_equal(sum_some.to_numpy(), [6, 8])

    def test_all(self):
        arr = TensorArray(np.arange(6).reshape(3, 2))
        s = pd.Series(arr)

        # Test as agg to TensorElement, defaults to axis=0
        result = s % 2 == 0
        npt.assert_array_equal(result.all(), np.array([True, False]))

    def test_any(self):
        arr = TensorArray(np.arange(6).reshape(3, 2))
        s = pd.Series(arr)

        # Test as agg to TensorElement, defaults to axis=0
        result = s % 3 == 0
        npt.assert_array_equal(result[2], np.array([False, False]))
        npt.assert_array_equal(result.any(), np.array([True, True]))

    def test_factorize(self):
        x = np.array([[1, 2], [3, 4], [5, 6], [3, 4]])
        s = TensorArray(x)
        with self.assertRaises(NotImplementedError):
            indices, values = s.factorize()

    def test_take(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)

        # Test no missing gets same dtype
        result = s.take([0, 2], allow_fill=True)
        expected = np.array([[1, 2], [5, 6]])
        self.assertEqual(result.numpy_dtype, expected.dtype)
        npt.assert_array_equal(result, expected)
        result = s.take([0, 2], allow_fill=False)
        npt.assert_array_equal(result, expected)

        # Test missing with nan fill gets promoted to float and filled
        result = s.take([1, -1], allow_fill=True)
        expected = np.array([[3, 4], [np.nan, np.nan]])
        self.assertEqual(result.numpy_dtype, expected.dtype)
        npt.assert_array_equal(result, expected)
        npt.assert_array_equal(result.isna(), [False, True])

    def test_numpy_properties(self):
        data = np.arange(6).reshape(3, 2)
        arr = TensorArray(data)
        self.assertEqual(arr.numpy_ndim, data.ndim)
        self.assertEqual(arr.numpy_shape, data.shape)
        self.assertEqual(arr.numpy_dtype, data.dtype)


class TensorArrayDataFrameTests(unittest.TestCase):
    def test_create(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        s = TensorArray(x)
        df = pd.DataFrame({"i": list(range(len(x))), "tensor": s})
        self.assertEqual(len(df), len(x))

    def test_sum(self):
        keys = ["a", "a", "b", "c", "c", "c"]
        values = np.array([[1, 1]] * len(keys))
        df = pd.DataFrame({"key": keys, "value": TensorArray(values)})
        result_df = df.groupby("key").aggregate({"value": "sum"})

        # Check array gets unwrapped from TensorElements
        arr = result_df["value"].array
        self.assertEqual(arr.numpy_dtype, values.dtype)
        npt.assert_array_equal(arr.to_numpy(), [[2, 2], [1, 1], [3, 3]])

        # Check the resulting DataFrame
        self.assertEqual(
            repr(result_df),
            textwrap.dedent(
                """\
                    value
                key      
                a   [2 2]
                b   [1 1]
                c   [3 3]"""
            ),
        )

        # 2D values
        values2 = np.array([[[1, 1], [1, 1]]] * len(keys))
        df2 = pd.DataFrame({"key": keys, "value": TensorArray(values2)})
        result2_df = df2.groupby("key").aggregate({"value": "sum"})

        # Check array gets unwrapped from TensorElements
        arr2 = result2_df["value"].array
        self.assertEqual(arr2.numpy_dtype, values.dtype)
        npt.assert_array_equal(arr2.to_numpy(),
                               [[[2, 2], [2, 2]], [[1, 1], [1, 1]], [[3, 3], [3, 3]]])

        # Check the resulting DataFrame
        self.assertEqual(
            repr(result2_df),
            textwrap.dedent(
                """\
                             value
                key               
                a   [[2 2]
                 [2 2]]
                b   [[1 1]
                 [1 1]]
                c   [[3 3]
                 [3 3]]"""
            ),
        )

    def test_bool_indexing_dataframe(self):
        s = TensorArray([[1, 2], [3, 4]])
        df = pd.DataFrame({
            "col1": s
        })
        result = df[[False, False]]
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), 0)

        result = df[[True, True]]
        self.assertTrue(isinstance(result, pd.DataFrame))
        pd.testing.assert_frame_equal(result, df)

        result = df[[True, False]]
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), 1)
        expected = df.iloc[[0]]
        pd.testing.assert_frame_equal(result, expected)

        result = df[[False, True]]
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), 1)
        expected = df.iloc[[1]]
        pd.testing.assert_frame_equal(result, expected)

    def test_bool_indexing_series(self):
        s = pd.Series(TensorArray([[1, 2], [3, 4]]))
        result = s[[False, False]]
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), 0)

        result = s[[True, True]]
        self.assertTrue(isinstance(result, pd.Series))
        pd.testing.assert_series_equal(result, s)

        result = s[[True, False]]
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), 1)
        expected = s.iloc[[0]]
        pd.testing.assert_series_equal(result, expected)

        result = s[[False, True]]
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), 1)
        expected = s.iloc[[1]]
        pd.testing.assert_series_equal(result, expected)

    def test_sort(self):
        arr = TensorArray(np.arange(6).reshape(3, 2))
        date_range = pd.date_range('2018-01-01', periods=3, freq='H')
        df = pd.DataFrame({"time": date_range, "tensor": arr})
        df = df.sort_values(by="time", ascending=False)
        self.assertEqual(df["tensor"].array.numpy_dtype, arr.numpy_dtype)
        expected = np.array([[4, 5], [2, 3], [0, 1]])
        npt.assert_array_equal(df["tensor"].array, expected)

    def test_large_display_numeric(self):

        # Test integer, uses IntArrayFormatter
        df = pd.DataFrame({"foo": TensorArray(np.array([[1, 2]] * 100))})
        self.assertEqual(
            repr(df),
            textwrap.dedent(
                """\
                     foo
                0  [1 2]
                1  [1 2]
                2  [1 2]
                3  [1 2]
                4  [1 2]
                ..   ...
                95 [1 2]
                96 [1 2]
                97 [1 2]
                98 [1 2]
                99 [1 2]
                
                [100 rows x 1 columns]"""
            )
        )

        # Test float, uses IntArrayFormatter
        df = pd.DataFrame({"foo": TensorArray(np.array([[1.1, 2.2]] * 100))})
        self.assertEqual(
            repr(df),
            textwrap.dedent(
                """\
                         foo
                0  [1.1 2.2]
                1  [1.1 2.2]
                2  [1.1 2.2]
                3  [1.1 2.2]
                4  [1.1 2.2]
                ..       ...
                95 [1.1 2.2]
                96 [1.1 2.2]
                97 [1.1 2.2]
                98 [1.1 2.2]
                99 [1.1 2.2]
                
                [100 rows x 1 columns]"""
            )
        )

    @pytest.mark.skipif(LooseVersion(pd.__version__) < LooseVersion("1.1.0"),
                        reason="Display of TensorArray with non-numeric dtype not supported")
    def test_large_display_string(self):

        # Uses the GenericArrayFormatter, doesn't work for Pandas 1.0.x but fixed in later versions
        df = pd.DataFrame({"foo": TensorArray(np.array([["Hello", "world"]] * 100))})
        self.assertEqual(
            repr(df),
            textwrap.dedent(
                """\
                                  foo
                0   ['Hello' 'world']
                1   ['Hello' 'world']
                2   ['Hello' 'world']
                3   ['Hello' 'world']
                4   ['Hello' 'world']
                ..                ...
                95  ['Hello' 'world']
                96  ['Hello' 'world']
                97  ['Hello' 'world']
                98  ['Hello' 'world']
                99  ['Hello' 'world']
                
                [100 rows x 1 columns]"""
            )
        )


class TensorArrayIOTests(unittest.TestCase):
    def test_feather(self):
        x = np.arange(10).reshape(5, 2)
        s = TensorArray(x)
        df = pd.DataFrame({"i": list(range(len(x))), "tensor": s})

        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, "tensor_array_test.feather")
            df.to_feather(filename)
            df_read = pd.read_feather(filename)
            pd.testing.assert_frame_equal(df, df_read)

    @pytest.mark.skipif(LooseVersion(pa.__version__) < LooseVersion("2.0.0"),
                        reason="Nested Parquet data types only supported in Arrow >= 2.0.0")
    def test_parquet(self):
        x = np.arange(10).reshape(5, 2)
        s = TensorArray(x)
        df = pd.DataFrame({"i": list(range(len(x))), "tensor": s})

        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, "tensor_array_test.parquet")
            df.to_parquet(filename)
            df_read = pd.read_parquet(filename)
            pd.testing.assert_frame_equal(df, df_read)

    def test_feather_chunked(self):
        from pyarrow.feather import write_feather

        x = np.arange(10).reshape(5, 2)
        s = TensorArray(x)
        df1 = pd.DataFrame({"i": list(range(len(s))), "tensor": s})

        # Create a Table with 2 chunks
        table1 = pa.Table.from_pandas(df1)
        df2 = df1.copy()
        df2["tensor"] = df2["tensor"] * 10
        table2 = pa.Table.from_pandas(df2)
        table = pa.concat_tables([table1, table2])
        self.assertEqual(table.column("tensor").num_chunks, 2)

        # Write table to feather and read back as a DataFrame
        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, "tensor_array_chunked_test.feather")
            write_feather(table, filename)
            df_read = pd.read_feather(filename)
            df_expected = pd.concat([df1, df2]).reset_index(drop=True)
            pd.testing.assert_frame_equal(df_expected, df_read)

    def test_feather_auto_chunked(self):
        from pyarrow.feather import read_table, write_feather

        x = np.arange(2048).reshape(1024, 2)
        s = TensorArray(x)
        df = pd.DataFrame({"i": list(range(len(s))), "tensor": s})

        table = pa.Table.from_pandas(df)

        # Write table to feather and read back as a DataFrame
        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, "tensor_array_chunked_test.feather")
            write_feather(table, filename, chunksize=512)
            table = read_table(filename)
            self.assertGreaterEqual(table.column("tensor").num_chunks, 2)
            df_read = pd.read_feather(filename)
            pd.testing.assert_frame_equal(df, df_read)


@pytest.fixture
def dtype():
    return TensorDtype()


@pytest.fixture
def data(dtype):
    values = np.array([[i] for i in range(100)])
    return pd.array(values, dtype=dtype)


@pytest.fixture
def data_for_twos(dtype):
    values = np.ones(100) * 2
    return pd.array(values, dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    values = np.array([[np.nan], [9]])
    return pd.array(values, dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    values = np.array([[2], [3], [1]])
    return pd.array(values, dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    values = np.array([[2], [np.nan], [1]])
    return pd.array(values, dtype=dtype)


@pytest.fixture
def na_cmp():
    return lambda x, y: (np.isnan(x) or np.all(np.isnan(x))) and \
                        (np.isnan(y) or np.all(np.isnan(y)))


@pytest.fixture
def na_value():
    return np.nan


@pytest.fixture
def data_for_grouping(dtype):
    a = [1]
    b = [2]
    c = [3]
    na = [np.nan]
    values = np.array([b, b, na, na, a, a, b, c])
    return pd.array(values, dtype=dtype)


# Can't import due to dependencies, taken from pandas.conftest import all_compare_operators
_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__"])
def all_compare_operators(request):
    return request.param


@pytest.fixture(params=["sum"])
def all_numeric_reductions(request):
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    return request.param

# import pytest fixtures
from pandas.tests.extension.conftest import all_data, as_array, as_frame, as_series, \
    box_in_series, data_repeated, fillna_method, groupby_apply_op, use_numpy


class TestPandasDtype(base.BaseDtypeTests):
    pass


class TestPandasInterface(base.BaseInterfaceTests):

    def test_array_interface(self, data):
        result = np.array(data)
        assert result[0] == data[0]

        # This invokes array interface of TensorArray
        result = np.array(data)
        assert result.dtype == np.int64

        # Must invoke array interface for each scalar
        expected = np.array([np.array(d) for d in data])
        npt.assert_array_equal(result, expected)


class TestPandasConstructors(base.BaseConstructorsTests):

    @pytest.mark.skip("using dtype=object unsupported")
    def test_pandas_array_dtype(self, data):
        # Fails making PandasArray with result = pd.array(data, dtype=np.dtype(object))
        pass


class TestPandasGetitem(base.BaseGetitemTests):

    def test_reindex(self, data, na_value):
        # Must make na_value same shape as data element
        na_value = np.array([na_value])
        super().test_reindex(data, na_value)


class TestPandasSetitem(base.BaseSetitemTests):

    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        mask = pd.array(np.zeros(data.shape, dtype="bool"), dtype="boolean")
        mask[:3] = True
        mask[3:5] = pd.NA

        if box_in_series:
            data = pd.Series(data)

        data[mask] = data[0]

        result = data[:3]
        if box_in_series:
            # Must unwrap Series
            result = result.values

        # Must compare all values of result
        assert np.all(result == data[0])


class TestPandasMissing(base.BaseMissingTests):
    @pytest.mark.skip(reason="TypeError: No matching signature found")
    def test_fillna_limit_pad(self, data_missing):
        super().test_fillna_limit_pad(data_missing)

    @pytest.mark.skip(reason="TypeError: No matching signature found")
    def test_fillna_limit_backfill(self, data_missing):
        super().test_fillna_limit_backfill(data_missing)

    @pytest.mark.skip(reason="TypeError: No matching signature found")
    def test_fillna_series_method(self, data_missing, fillna_method):
        super().test_fillna_series_method(data_missing, fillna_method)


class TestPandasArithmeticOps(base.BaseArithmeticOpsTests):

    # Expected errors for tests
    base.BaseArithmeticOpsTests.series_scalar_exc = None
    base.BaseArithmeticOpsTests.series_array_exc = None
    base.BaseArithmeticOpsTests.frame_scalar_exc = None
    base.BaseArithmeticOpsTests.divmod_exc = NotImplementedError

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        """ Override because creates Series from list of TensorElements as dtype=object."""
        # ndarray & other series
        op_name = all_arithmetic_operators
        s = pd.Series(data)
        self.check_opname(
            s, op_name, pd.Series([s.iloc[0]] * len(s), dtype=TensorDtype()), exc=self.series_array_exc
        )

    @pytest.mark.skip(reason="TensorArray does not error on ops")
    def test_error(self, data, all_arithmetic_operators):
        # other specific errors tested in the TensorArray specific tests
        pass


class TestPandasComparisonOps(base.BaseComparisonOpsTests):

    def _compare_other(self, s, data, op_name, other):
        """
        Override to eval result of `all()` as a `ndarray`.
        NOTE: test_compare_scalar uses value `0` for other.
        """
        op = self.get_op_from_name(op_name)
        if op_name == "__eq__":
            assert not np.all(op(s, other).all())
        elif op_name == "__ne__":
            assert np.all(op(s, other).all())
        elif op_name in ["__lt__", "__le__"]:
            assert not np.all(op(s, other).all())
        elif op_name in ["__gt__", "__ge__"]:
            assert np.all(op(s, other).all())
        else:
            raise ValueError("Unexpected opname: {}".format(op_name))

    def test_compare_scalar(self, data, all_compare_operators):
        """ Override to change scalar value to something usable."""
        op_name = all_compare_operators
        s = pd.Series(data)
        self._compare_other(s, data, op_name, -1)

    def test_compare_array(self, data, all_compare_operators):
        """ Override to change scalar value to something usable."""
        op_name = all_compare_operators
        s = pd.Series(data[1:])
        other = pd.Series([data[0]] * len(s), dtype=TensorDtype())
        self._compare_other(s, data, op_name, other)


@pytest.mark.skip("resolve errors")
class TestPandasReshaping(base.BaseReshapingTests):
    pass


@pytest.mark.skip("resolve errors")
class TestPandasMethods(base.BaseMethodsTests):
    pass


class TestPandasCasting(base.BaseCastingTests):
    pass


@pytest.mark.skip("resolve errors")
class TestPandasGroupby(base.BaseGroupbyTests):
    pass


@pytest.mark.skip("resolve errors")
class TestPandasNumericReduce(base.BaseNumericReduceTests):
    pass


@pytest.mark.skip("resolve errors")
class TestPandasBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestPandasPrinting(base.BasePrintingTests):

    @pytest.mark.skip("resolve errors")
    def test_array_repr(self, data, size):
        pass


class TestPandasUnaryOps(base.BaseUnaryOpsTests):

    @pytest.mark.skip("is supported?")
    def test_invert(self, data):
        pass


@pytest.mark.skip("Unsupported: must implement _from_sequence_of_strings")
class TestPandasParsing(base.BaseParsingTests):
    pass
