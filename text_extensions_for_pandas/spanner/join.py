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


def adjacent_join(first_series: pd.Series,
                  second_series: pd.Series,
                  first_name: str = "first",
                  second_name: str = "second",
                  min_gap: int = 0,
                  max_gap: int = 0):
    """
    Compute the join of two series of spans, where a pair of spans is
    considered to match if they are adjacent to each other in the text.

    :param first_series: Spans that appear earlier. dtype must be TokenSpan.

    :param second_series: Spans that come after. dtype must be TokenSpan.

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
    # below accordingly.
    outer = pd.DataFrame({
        "outer_span": first_series,
        "outer_end": first_series.values.end_token
    })

    # Inner series gets replicated for every possible offset so we can use
    # Pandas' high-performance equijoin
    inner_span_list = [second_series] * (max_gap - min_gap + 1)
    outer_end_list = [
        # Join predicate: outer_span = inner_span.begin + gap
        second_series.values.begin_token + gap
        for gap in range(min_gap, max_gap + 1)
    ]
    inner = pd.DataFrame({
        "inner_span": pd.concat(inner_span_list),
        "outer_end": np.concatenate(outer_end_list)
    })
    joined = outer.merge(inner)

    # Now we have a DataFrame with the schema
    # [outer_span, outer_end, inner_span]
    return pd.DataFrame({
        first_name: joined["outer_span"],
        second_name: joined["inner_span"]
    })
