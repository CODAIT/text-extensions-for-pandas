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
# join.py
#
# Span-specific join operators.
#

import numpy as np
import pandas as pd


def adjacent_join(
    first_series: pd.Series,
    second_series: pd.Series,
    first_name: str = "first",
    second_name: str = "second",
    min_gap: int = 0,
    max_gap: int = 0,
):
    """
    Compute the join of two series of spans, where a pair of spans is
    considered to match if they are adjacent to each other in the text.

    :param first_series: Spans that appear earlier. dtype must be TokenSpanDtype.

    :param second_series: Spans that come after. dtype must be TokenSpanDtype.

    :param first_name: Name to give the column in the returned dataframe that
    is derived from `first_series`.

    :param second_name: Column name for spans from `second_series` in the
    returned DataFrame.

    :param min_gap: Minimum number of spans allowed between matching pairs of
    spans, inclusive.

    :param max_gap: Maximum number of spans allowed between matching pairs of
    spans, inclusive.

    :returns: a new `pd.DataFrame` containing all pairs of spans that match
    the join predicate. Columns of the DataFrame will be named according
    to the `first_name` and `second_name` arguments.
    """
    # For now we always make the first series the outer.
    # TODO: Make the larger series the outer and adjust the join logic
    #  below accordingly.
    outer = pd.DataFrame(
        {"outer_span": first_series, "outer_end": first_series.values.end_token}
    )

    # Inner series gets replicated for every possible offset so we can use
    # Pandas' high-performance equijoin
    inner_span_list = [second_series] * (max_gap - min_gap + 1)
    outer_end_list = [
        # Outer comes first, so join predicate is:
        #     outer_span.end + gap == inner_span.begin
        # or equivalently:
        #     outer_span.end = inner_span.begin - gap
        second_series.values.begin_token - gap
        for gap in range(min_gap, max_gap + 1)
    ]
    inner = pd.DataFrame(
        {
            "inner_span": pd.concat(inner_span_list),
            "outer_end": np.concatenate(outer_end_list),
        }
    )
    joined = outer.merge(inner)

    # Now we have a DataFrame with the schema
    # [outer_span, outer_end, inner_span]
    return pd.DataFrame(
        {first_name: joined["outer_span"], second_name: joined["inner_span"]}
    )


def overlap_join(
    first_series: pd.Series,
    second_series: pd.Series,
    first_name: str = "first",
    second_name: str = "second",
):
    """
    Compute the join of two series of spans, where a pair of spans is
    considered to match if they overlap.

    :param first_series: First set of spans to join, wrapped in a `pd.Series`

    :param second_series: Second set of spans to join.

    :param first_name: Name to give the column in the returned dataframe that
    is derived from `first_series`.

    :param second_name: Column name for spans from `second_series` in the
    returned DataFrame.

    :returns: a new `pd.DataFrame` containing all pairs of spans that match
    the join predicate. Columns of the DataFrame will be named according
    to the `first_name` and `second_name` arguments.
    """
    # For now we always use character offsets.
    # TODO: Use token offsets of both sides of the join are TokenSpanArrays
    def _get_char_offsets(s: pd.Series):
        # noinspection PyUnresolvedReferences
        return s.values.begin, s.values.end

    first_begins, first_ends = _get_char_offsets(first_series)
    second_begins, second_ends = _get_char_offsets(second_series)

    # The algorithm here is what is known in the ER literature as "blocking".
    # First evaluate a looser predicate that can be translated to an equijoin,
    # then filter using the actual join predicate.

    # Compute average span length to determine blocking factor
    # TODO: Is average the right aggregate to use here?
    total_len = np.sum(first_ends - first_begins) + np.sum(second_ends - second_begins)
    average_len = total_len / (len(first_series) + len(second_series))
    blocking_factor = max(1, int(np.floor(average_len)))

    # Generate a table of which blocks each row of the input participates in.
    # Use primary key (index) values because inputs can have duplicate spans.
    def _make_table(name, index, begins, ends):
        # TODO: Vectorize this part.
        indexes = []
        blocks = []
        for i, b, e in zip(index, begins, ends):
            for block in range(b // blocking_factor, e // blocking_factor + 1):
                indexes.append(i)
                blocks.append(block)
        return pd.DataFrame({name: indexes, "block": blocks})

    first_table = _make_table("first", first_series.index, first_begins, first_ends)
    second_table = _make_table(
        "second", second_series.index, second_begins, second_ends
    )

    # Do an equijoin on block ID and remove duplicates from the resulting
    # <first key, second key> relation.
    merged_table = pd.merge(first_table, second_table)
    key_pairs = merged_table.groupby(["first", "second"]).aggregate(
        {"first": "first", "second": "first"}
    )

    # Join the keys back with the original series to form the result, plus
    # some extra values due to blocking.
    block_result = pd.DataFrame(
        {
            first_name: first_series.loc[key_pairs["first"]].values,
            second_name: second_series.loc[key_pairs["second"]].values,
        }
    )

    # Filter out extra values from blocking
    mask = block_result[first_name].values.overlaps(block_result[second_name].values)
    return block_result[mask].reset_index(drop=True)


def contain_join(
    first_series: pd.Series,
    second_series: pd.Series,
    first_name: str = "first",
    second_name: str = "second",
):
    """
    Compute the join of two series of spans, where a pair of spans is
    considered to match if the second span is contained within the first.

    :param first_series: First set of spans to join, wrapped in a `pd.Series`

    :param second_series: Second set of spans to join. These are the ones that
     are contained within the first set where the join predicate is satisfied.

    :param first_name: Name to give the column in the returned dataframe that
    is derived from `first_series`.

    :param second_name: Column name for spans from `second_series` in the
    returned DataFrame.

    :returns: a new `pd.DataFrame` containing all pairs of spans that match
    the join predicate. Columns of the DataFrame will be named according
    to the `first_name` and `second_name` arguments.
    """
    # For now we just run overlap_join() and filter the results.
    # TODO: Factor out the blocking code so that we can avoid filtering
    #  and regenerating the index twice.
    overlap_result = overlap_join(first_series, second_series, first_name, second_name)
    mask = overlap_result[first_name].values.contains(
        overlap_result[second_name].values
    )
    return overlap_result[mask].reset_index(drop=True)
