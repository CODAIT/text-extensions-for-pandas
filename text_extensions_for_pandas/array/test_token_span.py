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
# noinspection PyPackageRequirements
import pytest

from pandas.tests.extension import base

from text_extensions_for_pandas.array.test_span import ArrayTestBase
from text_extensions_for_pandas.array.span import *
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

        toks2 = SpanArray(
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
        s5 = Span(toks.target_text, s4.begin, s4.end)
        s6 = Span(toks.target_text, s4.begin, s4.end + 1)

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
        char_s1 = Span(s1.target_text, s1.begin, s1.end)
        char_s2 = Span(s2.target_text, s2.begin, s2.end)

        self.assertEqual(s1 + s2, s1)
        self.assertEqual(char_s1 + s2, char_s1)
        self.assertEqual(s2 + char_s1, char_s1)
        self.assertEqual(char_s2 + char_s1, char_s1)
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
        self.assertTrue(isinstance(arr.dtype, TokenSpanDtype))

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
        arr3 = SpanArray(arr.target_text, arr.begin, arr.end)
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
        char_s1 = Span(s1.target_text, s1.begin, s1.end)
        char_s2 = Span(s2.target_text, s2.begin, s2.end)
        char_s3 = Span(s3.target_text, s3.begin, s3.end)
        char_s4 = Span(s4.target_text, s4.begin, s4.end)
        char_s5 = Span(s5.target_text, s5.begin, s5.end)

        # TokenSpanArray + TokenSpanArray
        self._assertArrayEquals(
            TokenSpanArray._from_sequence([s1, s2, s3])
            + TokenSpanArray._from_sequence([s2, s3, s3]),
            TokenSpanArray._from_sequence([s1, s4, s3]),
        )
        # SpanArray + TokenSpanArray
        self._assertArrayEquals(
            SpanArray._from_sequence([char_s1, char_s2, char_s3])
            + TokenSpanArray._from_sequence([s2, s3, s3]),
            SpanArray._from_sequence([char_s1, char_s4, char_s3]),
        )
        # TokenSpanArray + SpanArray
        self._assertArrayEquals(
            TokenSpanArray._from_sequence([s1, s2, s3])
            + SpanArray._from_sequence([char_s2, char_s3, char_s3]),
            SpanArray._from_sequence([char_s1, char_s4, char_s3]),
        )
        # TokenSpanArray + TokenSpan
        self._assertArrayEquals(
            TokenSpanArray._from_sequence([s1, s2, s3]) + s2,
            TokenSpanArray._from_sequence([s5, s2, s4]),
        )
        # TokenSpan + TokenSpanArray
        self._assertArrayEquals(
            s2 + TokenSpanArray._from_sequence([s1, s2, s3]),
            TokenSpanArray._from_sequence([s5, s2, s4]),
        )
        # TokenSpanArray + Span
        self._assertArrayEquals(
            TokenSpanArray._from_sequence([s1, s2, s3]) + char_s2,
            SpanArray._from_sequence([char_s5, char_s2, char_s4]),
        )
        # Span + SpanArray
        self._assertArrayEquals(
            char_s2 + SpanArray._from_sequence([char_s1, char_s2, char_s3]),
            SpanArray._from_sequence([char_s5, char_s2, char_s4]),
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

        # With a Span column, TokenSpan padded to same length
        df5 = pd.DataFrame({"cs": toks})
        df5 = pd.concat([df3, df5], axis=1)
        self.do_roundtrip(df5)

        # All columns together, TokenSpan arrays padded as needed
        df = pd.concat([df1, df2, df3, df4], axis=1)
        self.do_roundtrip(df)


@pytest.fixture
def dtype():
    return TokenSpanDtype()


def _gen_spans():
    text = "1"
    begins = [0]
    ends = [1]
    for i in range(1, 100):
        s = str(i * 11)
        text += f" {s}"
        begins.append(ends[i - 1] + 1)
        ends.append(begins[i] + len(s))
    char_spans = [Span(text, b, e) for b, e in zip(begins, ends)]
    char_span_arr = pd.array(char_spans, dtype=SpanDtype())
    return (TokenSpan(char_span_arr, i, i + 1) for i in range(len(char_spans)))


@pytest.fixture
def data(dtype):
    return pd.array(list(_gen_spans()), dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    spans = [span for span, _ in zip(_gen_spans(), range(2))]
    spans[0] = TokenSpan(
        spans[0].tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE
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
    reordered[1] = TokenSpan(
        spans[0].tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE
    )
    reordered[2] = spans[1]
    return pd.array(reordered, dtype=dtype)


@pytest.fixture
def na_cmp():
    return lambda x, y: x.begin == TokenSpan.NULL_OFFSET_VALUE and \
                        y.begin == TokenSpan.NULL_OFFSET_VALUE


@pytest.fixture
def na_value():
    spans = [span for span, _ in zip(_gen_spans(), range(1))]
    return TokenSpan(spans[0].tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE)


@pytest.fixture
def data_for_grouping(dtype):
    spans = [span for span, _ in zip(_gen_spans(), range(3))]
    a = spans[0]
    b = spans[1]
    c = spans[2]
    na = TokenSpan(
        spans[0].tokens, TokenSpan.NULL_OFFSET_VALUE, TokenSpan.NULL_OFFSET_VALUE
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

    def test_construct_empty_dataframe(self, dtype):
        try:
            with pytest.raises(TypeError, match="Expected SpanArray as tokens"):
                super().test_construct_empty_dataframe(dtype)
        except AttributeError:
            # Test added in Pandas 1.1.0, ignore for earlier versions
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

    def test_factorize_empty(self, data):
        with pytest.raises(TypeError, match="Expected SpanArray"):
            super().test_factorize_empty(data)

    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy):
        if repeats == 0:
            # Leads to empty array, unsupported???
            with pytest.raises(TypeError, match="Expected SpanArray"):
                super().test_repeat(data, repeats, as_series, use_numpy)
        else:
            super().test_repeat(data, repeats, as_series, use_numpy)


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
        expected = TokenSpan(first.tokens, first.begin_token, last.end_token)
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
