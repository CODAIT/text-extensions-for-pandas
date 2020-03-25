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

################################################################################
# conll.py
#
# I/O functions related to CONLL entity format and its many derivatives.

import pandas as pd
import numpy as np

from text_extensions_for_pandas.array import (
    CharSpanArray,
    TokenSpanArray,
    CharSpanType,
    TokenSpanType,
)


def iob_to_spans(
    token_features: pd.DataFrame,
    iob_col_name: str = "ent_iob",
    char_span_col_name: str = "char_span",
    entity_type_col_name: str = "ent_type",
):
    """
    Convert token tags in Inside–Outside–Beginning (IOB) format to a series of
    `TokenSpan`s of entities.
    :param token_features: DataFrame of token features in the format returned by
     `make_tokens_and_features`.
    :param iob_col_name: Name of a column in `token_features` that contains the
     IOB tags as strings, "I", "O", or "B".
    :param char_span_col_name: Name of a column in `token_features` that
     contains the tokens as a `CharSpanArray`.
    :param entity_type_col_name: Optional name of a column in `token_features`
     that contains entity type information; or `None` if no such column exists.
    :return: A `pd.DataFrame` with the following columns:
    * `token_span`: Span (with token offsets) of each entity
    * `<value of entity_type_col_name>`: (optional) Entity type
    """
    # Start out with 1-token prefixes of all entities.
    begin_mask = token_features[iob_col_name] == "B"
    first_tokens = token_features[begin_mask].index
    if entity_type_col_name is None:
        entity_types = np.zeros(len(first_tokens))
    else:
        entity_types = token_features[begin_mask][entity_type_col_name]

    # Add an extra "O" tag to the end of the IOB column to simplify the logic
    # for handling the case where the document ends with an entity.
    iob_series = (
        token_features[iob_col_name].append(pd.Series(["O"])).reset_index(drop=True)
    )

    entity_prefixes = pd.DataFrame(
        {
            "ent_type": entity_types,
            "begin": first_tokens,  # Inclusive
            "end": first_tokens + 1,  # Exclusive
            "next_tag": iob_series.iloc[first_tokens + 1].values,
        }
    )

    df_list = []  # Type: pd.DataFrame

    if len(entity_prefixes.index) == 0:
        # Code below needs at least one element in the list for schema
        df_list = [entity_prefixes]

    # Iteratively expand the prefixes
    while len(entity_prefixes.index) > 0:
        complete_mask = entity_prefixes["next_tag"].isin(["O", "B"])
        complete_entities = entity_prefixes[complete_mask]
        incomplete_entities = entity_prefixes[~complete_mask].copy()
        incomplete_entities["end"] = incomplete_entities["end"] + 1
        incomplete_entities["next_tag"] = iob_series.iloc[
            incomplete_entities["end"]
        ].values
        df_list.append(complete_entities)
        entity_prefixes = incomplete_entities
    all_entities = pd.concat(df_list)

    # Sort spans by location, not length.
    all_entities.sort_values("begin", inplace=True)

    # Convert [begin, end) pairs to spans
    entity_spans_array = TokenSpanArray(
        token_features[char_span_col_name].values,
        all_entities["begin"].values,
        all_entities["end"].values,
    )
    if entity_type_col_name is None:
        return pd.DataFrame({"token_span": entity_spans_array})
    else:
        return pd.DataFrame(
            {
                "token_span": entity_spans_array,
                entity_type_col_name: all_entities["ent_type"].values,
            }
        )
