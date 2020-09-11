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
# consolidate.py
#
# Variants of the Consolidate operator from spanner algebra.
# The Consolidate operator removes spans that conflict with other spans
# according to a consolidation policy.
#

import pandas as pd

from text_extensions_for_pandas.array.span import SpanArray


def consolidate(df: pd.DataFrame, on: str, how: str = "left_to_right") -> pd.DataFrame:
    """
    Eliminate overlap among the spans in one column of a
    :class:`pd.DataFrame`.

    :param df: DataFrame containing spans and other attributes
    :param on: Name of column in `df` on which to perform consolidation
    :param how: What policy to use to decide what spans are considered
     to overlap and which of an overlapping pair will remain after
     consolidation. Available policies:
     * `left_to_right`: Walk through the spans from left to right, keeping
       the longest non-overlapping match at each position encountered

    Returns the rows of `df` that remain after applying the specified
    policy to the spans in the column specified by `on`.
    """
    spans = df[on].values
    if not isinstance(spans, SpanArray):
        raise TypeError(f"Column '{on}' of dataframe is of type "
                        f"{df[on].dtype}, which is not a span type.")
    if how != "left_to_right":
        raise ValueError(f"Receieved '{how}' for `how` argument, but "
                         f"the only valid value for that argument is "
                         f"'left_to_right'.")

    tmp = pd.DataFrame({
        "span": spans,
        "begin": spans.begin,
        "end": spans.end,
        "ix": range(len(spans))}
    ).sort_values(["begin", "end"], ascending=[True, False])

    # Slow-but-correct implementation for now
    ix_to_retain = []  # Type: List[int]
    iloc = 0

    while iloc < len(tmp.index):
        # Loop invariants:
        # * iloc == location of a span that doesn't overlap with any
        #           span in ix_to_retain
        # * All locations before iloc have been processed.

        # Since we sorted by end in DESCENDING order, the current span
        # is guaranteed to be the longest span that begins at its begin
        # offset.
        row = tmp.iloc[iloc]
        cur_end = row["end"]
        cur_ix = row["ix"]
        ix_to_retain.append(cur_ix)

        # Skip other spans that begin before this span ends
        while (iloc < len(tmp.index)
               and tmp.iloc[iloc]["begin"] < cur_end):
            iloc += 1

    return df.iloc[ix_to_retain]