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

import pandas as pd
import os
import tempfile
import unittest

from pandas.tests.extension import base
# noinspection PyPackageRequirements
import pytest

from text_extensions_for_pandas.array.span import *

from text_extensions_for_pandas.util import TestBase


class SpanTest(unittest.TestCase):
    def test_create(self):
        test_text = "This is a test."
        s1 = Span(test_text, 0, 4)
        self.assertEqual(s1.covered_text, "This")

        # Begin too small
        with self.assertRaises(ValueError):
            Span(test_text, -2, 4)

        # End too small
        with self.assertRaises(ValueError):
            Span(test_text, 1, -1)

        # Begin null, end not null
        with self.assertRaises(ValueError):
            Span(test_text, Span.NULL_OFFSET_VALUE, 0)

    def test_repr(self):
        test_text = "This is a test."
        s1 = Span(test_text, 0, 4)
        self.assertEqual(repr(s1), "[0, 4): 'This'")

    def test_equals(self):
        test_text = "This is a test."
        s1 = Span(test_text, 0, 4)
        s2 = Span(test_text, 0, 4)
        s3 = Span(test_text, 0, 5)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_less_than(self):
        test_text = "This is a test."
        s1 = Span(test_text, 0, 4)
        s2 = Span(test_text, 2, 4)
        s3 = Span(test_text, 4, 5)

        self.assertLess(s1, s3)
        self.assertLessEqual(s1, s3)
        self.assertFalse(s1 < s2)

    def test_overlaps(self):
        test_text = "This is a test."
        s1 = Span(test_text, 2, 4)
        no_overlap = [
            Span(test_text, 0, 1),
            Span(test_text, 0, 2),
            Span(test_text, 1, 1),
            Span(test_text, 2, 2),
            Span(test_text, 4, 4),
            Span(test_text, 4, 5),
            Span(test_text, 5, 7),
        ]
        overlap = [
            Span(test_text, 1, 3),
            Span(test_text, 1, 4),
            Span(test_text, 2, 3),
            Span(test_text, 2, 4),
            Span(test_text, 2, 5),
            Span(test_text, 3, 3),
            Span(test_text, 3, 4),
        ]

        for s_other in no_overlap:
            self.assertFalse(s1.overlaps(s_other))
            self.assertFalse(s_other.overlaps(s1))

        for s_other in overlap:
            self.assertTrue(s1.overlaps(s_other))
            self.assertTrue(s_other.overlaps(s1))

        s2 = Span(test_text, 1, 1)
        s3 = Span(test_text, 1, 1)
        s4 = Span(test_text, 2, 2)
        self.assertTrue(s2.overlaps(s3))
        self.assertFalse(s3.overlaps(s4))

    def test_contains(self):
        test_text = "This is a test."
        s1 = Span(test_text, 2, 4)
        not_contained = [
            Span(test_text, 0, 1),
            Span(test_text, 0, 2),
            Span(test_text, 1, 1),
            Span(test_text, 1, 3),
            Span(test_text, 1, 4),
            Span(test_text, 2, 5),
            Span(test_text, 4, 5),
            Span(test_text, 5, 7),
        ]
        contained = [
            Span(test_text, 2, 2),
            Span(test_text, 2, 3),
            Span(test_text, 2, 4),
            Span(test_text, 3, 3),
            Span(test_text, 3, 4),
            Span(test_text, 4, 4),
        ]

        for s_other in not_contained:
            self.assertFalse(s1.contains(s_other))

        for s_other in contained:
            self.assertTrue(s1.contains(s_other))

        s2 = Span(test_text, 1, 1)
        s3 = Span(test_text, 1, 1)
        s4 = Span(test_text, 2, 2)
        self.assertTrue(s2.contains(s3))
        self.assertFalse(s3.contains(s4))

    def test_context(self):
        test_text = "This is a test."
        s1 = Span(test_text, 5, 7)
        s2 = Span(test_text, 0, 4)
        s3 = Span(test_text, 10, 15)

        self.assertEqual(s1.context(), "This [is] a test.")
        self.assertEqual(s1.context(2), "...s [is] a...")
        self.assertEqual(s2.context(), "[This] is a test.")
        self.assertEqual(s2.context(3), "[This] is...")
        self.assertEqual(s3.context(), "This is a [test.]")
        self.assertEqual(s3.context(3), "... a [test.]")

    def test_hash(self):
        test_text = "This is a test."
        s1 = Span(test_text, 0, 3)
        s2 = Span(test_text, 0, 3)
        s3 = Span(test_text, 3, 4)
        d = {s1: "foo"}
        self.assertEqual(d[s1], "foo")
        self.assertEqual(d[s2], "foo")
        d[s2] = "bar"
        d[s3] = "fab"
        self.assertEqual(d[s1], "bar")
        self.assertEqual(d[s2], "bar")
        self.assertEqual(d[s3], "fab")


class ArrayTestBase(TestBase):
    """
    Shared base class for CharSpanArrayTest and TokenSpanArrayTest
    """

    @staticmethod
    def _make_spans_of_tokens():
        """
        :return: An example SpanArray containing the tokens of the string
          "This is a test.", not including the period at the end.
        """
        return SpanArray(
            "This is a test.", np.array([0, 5, 8, 10]), np.array([4, 7, 9, 14])
        )


class CharSpanArrayTest(ArrayTestBase):
    def test_create(self):
        arr = self._make_spans_of_tokens()
        self._assertArrayEquals(arr.covered_text, ["This", "is", "a", "test"])

        with self.assertRaises(TypeError):
            SpanArray("", "Not a valid begins list", [42])

    def test_dtype(self):
        arr = SpanArray("", np.array([0], ), np.array([0]))
        self.assertTrue(isinstance(arr.dtype, SpanDtype))

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
        arr[2] = Span(
            arr.target_text, Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE
        )
        self.assertIsNone(arr.covered_text[2])
        self._assertArrayEquals(arr.covered_text, ["This", "is", None, "test"])
        self._assertArrayEquals(arr.isna(), [False, False, True, False])

    def test_copy(self):
        arr = self._make_spans_of_tokens()
        arr2 = arr.copy()
        arr[0] = Span(arr.target_text, 8, 9)
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
        arr1 = SpanArray(
            "This is a test.", np.array([0, 5, 8, 10]), np.array([4, 7, 9, 14])
        )
        s1 = Span(arr1.target_text, 0, 1)
        s2 = Span(arr1.target_text, 11, 14)
        arr2 = SpanArray(arr1.target_text, [0, 3, 10, 7], [0, 4, 12, 9])

        self._assertArrayEquals(s1 < arr1, [False, True, True, True])
        self._assertArrayEquals(s2 > arr1, [True, True, True, False])
        self._assertArrayEquals(arr1 < s1, [False, False, False, False])
        self._assertArrayEquals(arr1 < arr2, [False, False, True, False])

    def test_reduce(self):
        arr = self._make_spans_of_tokens()
        self.assertEqual(arr._reduce("sum"), Span(arr.target_text, 0, 14))
        # Remind ourselves to modify this test after implementing min and max
        with self.assertRaises(TypeError):
            arr._reduce("min")

    def test_as_tuples(self):
        arr = SpanArray(
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
        s1 = Span(test_text, 2, 4)
        s_others = [
            # Non-overlapping
            Span(test_text, 0, 1),
            Span(test_text, 0, 2),
            Span(test_text, 1, 1),
            Span(test_text, 2, 2),
            Span(test_text, 4, 4),
            Span(test_text, 4, 5),
            Span(test_text, 5, 7),
            # Overlapping
            Span(test_text, 1, 3),
            Span(test_text, 1, 4),
            Span(test_text, 2, 3),
            Span(test_text, 2, 4),
            Span(test_text, 2, 5),
            Span(test_text, 3, 3),
            Span(test_text, 3, 4),
        ]

        expected_mask = [False] * 7 + [True] * 7

        s1_array = SpanArray._from_sequence([s1] * len(s_others))
        others_array = SpanArray._from_sequence(s_others)

        self._assertArrayEquals(s1_array.overlaps(others_array), expected_mask)
        self._assertArrayEquals(others_array.overlaps(s1_array), expected_mask)
        self._assertArrayEquals(others_array.overlaps(s1), expected_mask)

        # Check zero-length span cases
        one_one = SpanArray._from_sequence([Span(test_text, 1, 1)] * 2)
        one_one_2_2 = SpanArray._from_sequence(
            [Span(test_text, 1, 1), Span(test_text, 2, 2)]
        )
        self._assertArrayEquals(one_one.overlaps(one_one_2_2), [True, False])
        self._assertArrayEquals(one_one_2_2.overlaps(one_one), [True, False])
        self._assertArrayEquals(
            one_one_2_2.overlaps(Span(test_text, 1, 1)), [True, False]
        )

    def test_contains(self):
        test_text = "This is a test."
        s1 = Span(test_text, 2, 4)
        s_others = [
            # Not contained within s1
            Span(test_text, 0, 1),
            Span(test_text, 0, 2),
            Span(test_text, 1, 1),
            Span(test_text, 1, 3),
            Span(test_text, 1, 4),
            Span(test_text, 2, 5),
            Span(test_text, 4, 5),
            Span(test_text, 5, 7),
            # Contained within s1
            Span(test_text, 2, 2),
            Span(test_text, 2, 3),
            Span(test_text, 2, 4),
            Span(test_text, 3, 3),
            Span(test_text, 3, 4),
            Span(test_text, 4, 4),
        ]

        expected_mask = [False] * 8 + [True] * 6

        s1_array = SpanArray._from_sequence([s1] * len(s_others))
        others_array = SpanArray._from_sequence(s_others)

        self._assertArrayEquals(s1_array.contains(others_array), expected_mask)

        # Check zero-length span cases
        one_one = SpanArray._from_sequence([Span(test_text, 1, 1)] * 2)
        one_one_2_2 = SpanArray._from_sequence(
            [Span(test_text, 1, 1), Span(test_text, 2, 2)]
        )
        self._assertArrayEquals(one_one.contains(one_one_2_2), [True, False])
        self._assertArrayEquals(one_one_2_2.contains(one_one), [True, False])
        self._assertArrayEquals(
            one_one_2_2.contains(Span(test_text, 1, 1)), [True, False]
        )


class CharSpanArrayIOTests(ArrayTestBase):

    def test_feather(self):
        arr = self._make_spans_of_tokens()
        df = pd.DataFrame({'Span': arr})

        with tempfile.TemporaryDirectory() as dirpath:
            filename = os.path.join(dirpath, 'char_span_array_test.feather')
            df.to_feather(filename)
            df_read = pd.read_feather(filename)
            pd.testing.assert_frame_equal(df, df_read)


@pytest.fixture
def dtype():
    return SpanDtype()


def _gen_spans():
    text = "1"
    begins = [0]
    ends = [1]
    for i in range(1, 100):
        s = str(i * 11)
        text += f" {s}"
        begins.append(ends[i - 1] + 1)
        ends.append(begins[i] + len(s))
    return (Span(text, b, e) for b, e in zip(begins, ends))


@pytest.fixture
def data(dtype):
    return pd.array(list(_gen_spans()), dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    spans = [span for span, _ in zip(_gen_spans(), range(2))]
    spans[0] = Span(
        spans[0].target_text, Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE
    )
    return pd.array(spans, dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    spans = [span for span, _ in zip(_gen_spans(), range(3))]
    reordered = [None] * len(spans)
    reordered[0] = spans[1]
    reordered[1] = spans[2]
    reordered[2] = spans[0]
    return pd.array(reordered, dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    spans = [span for span, _ in zip(_gen_spans(), range(3))]
    reordered = [None] * len(spans)
    reordered[0] = spans[2]
    reordered[1] = Span(
        spans[0].target_text, Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE
    )
    reordered[2] = spans[1]
    return pd.array(reordered, dtype=dtype)


@pytest.fixture
def na_cmp():
    return lambda x, y: x.begin == Span.NULL_OFFSET_VALUE and \
                        y.begin == Span.NULL_OFFSET_VALUE


@pytest.fixture
def na_value():
    spans = [span for span, _ in zip(_gen_spans(), range(1))]
    return Span(spans[0].target_text, Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE)


@pytest.fixture
def data_for_grouping(dtype):
    spans = [span for span, _ in zip(_gen_spans(), range(3))]
    a = spans[0]
    b = spans[1]
    c = spans[2]
    na = Span(
        spans[0].target_text, Span.NULL_OFFSET_VALUE, Span.NULL_OFFSET_VALUE
    )
    return pd.array([b, b, na, na, a, a, b, c], dtype=dtype)


# Can't import due to dependencies, taken from pandas.conftest import all_compare_operators
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
    pass


class TestPandasConstructors(base.BaseConstructorsTests):

    @pytest.mark.skip("Unsupported, sequence of all NaNs")
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        pass


class TestPandasGetitem(base.BaseGetitemTests):
    pass


class TestPandasSetitem(base.BaseSetitemTests):
    pass


class TestPandasMissing(base.BaseMissingTests):
    pass


@pytest.mark.skip("not applicable")
class TestPandasArithmeticOps(base.BaseArithmeticOpsTests):
    pass


class TestPandasComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        if op_name in ["__le__", "__ge__"]:
            pytest.skip("op not supported")
        op = self.get_op_from_name(op_name)
        if isinstance(other, int):
            # Compare with other type of object
            with pytest.raises(ValueError):
                op(data, other)

            # Compare with scalar
            other = data[0]

        # TODO check result
        op(data, other)

    @pytest.mark.skip("assert result is NotImplemented")
    def test_direct_arith_with_series_returns_not_implemented(self, data):
        pass


class TestPandasReshaping(base.BaseReshapingTests):
    @pytest.mark.skip(reason="resolve errors")
    def test_unstack(self, data, index, obj):
        pass

    @pytest.mark.skip(reason="ValueError: Spans must all be over the same target text")
    def test_concat_with_reindex(self, data):
        pass


class TestPandasMethods(base.BaseMethodsTests):
    @pytest.mark.skip(reason="Unclear test")
    def test_value_counts(self, all_data, dropna):
        pass

    @pytest.mark.skip(reason="invalid operator")
    def test_combine_add(self, data_repeated):
        pass

    @pytest.mark.skip(reason="unsupported operation")
    def test_container_shift(self, data, frame, periods, indices):
        # TODO check if support required
        pass

    @pytest.mark.skip(reason="unsupported operation")
    def test_shift_non_empty_array(self, data, periods, indices):
        # TODO check if support required
        pass

    @pytest.mark.skip(reason="unsupported operation")
    def test_where_series(self, data, na_value, as_frame):
        # TODO setitem error: NotImplementedError: Setting multiple rows at once not implemented
        pass

    def test_searchsorted(self, data_for_sorting, as_series):
        # TODO fails for series with TypeError: 'Span' object is not iterable
        if as_series is True:
            pytest.skip("errors with Series")
        super().test_searchsorted(data_for_sorting, as_series)

    @pytest.mark.skip("AttributeError: 'SpanArray' object has no attribute 'value_counts'")
    def test_value_counts_with_normalize(self, data):
        pass

    @pytest.mark.skip("Failed: DID NOT RAISE <class 'TypeError'>")
    def test_not_hashable(self, data):
        pass

    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        from pandas.core.dtypes.generic import ABCPandasArray
        if isinstance(box, ABCPandasArray):
            pytest.skip("TypeError: equals() not defined for arguments of type <class 'NoneType'>")


class TestPandasCasting(base.BaseCastingTests):
    pass


class TestPandasGroupby(base.BaseGroupbyTests):
    pass


class TestPandasNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, s, op_name, skipna):
        # TODO skipna has no bearing
        result = getattr(s, op_name)(skipna=skipna)
        first = s[0]
        last = s[len(s) - 1]
        expected = Span(first.target_text, first.begin, last.end)
        assert result == expected


@pytest.mark.skip("must support 'all', 'any' aggregations")
class TestPandasBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestPandasPrinting(base.BasePrintingTests):
    pass


class TestPandasUnaryOps(base.BaseUnaryOpsTests):

    @pytest.mark.skip("is supported?")
    def test_invert(self, data):
        pass


@pytest.mark.skip("must implement _from_sequence_of_strings")
class TestPandasParsing(base.BaseParsingTests):
    pass


if __name__ == "__main__":
    unittest.main()
