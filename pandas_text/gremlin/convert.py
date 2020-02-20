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
import json
import textwrap
from typing import Iterable

import pandas as pd

from pandas_text.char_span import CharSpan
from pandas_text.gremlin.traversal.constant import PrecomputedTraversal

# convert.py
#
# Functions for converting dataframes to Gremlin traversal objects


def token_features_to_traversal(token_features: pd.DataFrame,
                                drop_self_links: bool = True,
                                link_cols: Iterable[str] = (
                                    "head", "left", "right")):
    """
    Turn a DataFrame of token features in the form returned by
    `make_tokens_and_features` into an empty graph traversal.
    Similar to calling `graph.traversal()` in Gremlin.

    :param token_features: DataFrame containing information about individual
    tokens. Must contain a `head_token_num` column that is synchronized with
    the DataFrame's index.

    :param drop_self_links: If `True`, remove links from nodes to themselves
    to simplify query logic.

    :param link_cols: Names of the columns to treat as links, if present.
     This function will ignore any name in this list that doesn't match a
     column name in `token_features`.

    :returns: A traversal containing a graph version of `token_features` and
    an empty set of paths.
    """
    valid_link_cols = set(link_cols).intersection(token_features.columns)
    vertices = token_features.copy()  # Make a shallow copy just in case
    # Add edges for every column name in link_cols that is present.
    edges_list = []
    for name in valid_link_cols:
        df = pd.DataFrame(
            {"from": token_features.index, "to": token_features[name],
             "type": name})
        edges_list.append(df[~df["to"].isnull()])
    edges = pd.concat(edges_list)
    if drop_self_links:
        edges = edges[edges["from"] != edges["to"]]
    paths = pd.DataFrame()
    step_types = []
    aliases = {}
    return PrecomputedTraversal(vertices, edges, paths, step_types, aliases)


def token_features_to_gremlin(token_features: pd.DataFrame,
                              drop_self_links=True,
                              include_begin_and_end=False):
    """
    :param token_features: A subset of a token features DataFrame in the format
    returned by `make_tokens_and_features()`.

    :param drop_self_links: If `True`, remove links from nodes to themselves
    to simplify query logic.

    :param include_begin_and_end: If `True`, break out the begin and end
    attributes of spans as separate fields.

    :return: A string of Gremlin commands that you can paste into the Gremlin
    console to generate a graph that models the contents of `token_features`.
    """

    def _quote_str(v):
        return json.dumps(str(v))

    # Nodes:
    # For each token, generate addV("token").property("key","value")...as(id)
    node_lines = []
    colnames = token_features.columns
    for row in token_features.itertuples(index=True):
        # First element in tuple is index value
        index_val = row[0]
        props_list = []
        for i in range(len(colnames)):
            cell = row[i + 1]
            props_list.append(
                ".property({}, {})".format(
                    _quote_str(colnames[i]), _quote_str(cell))
            )
            if include_begin_and_end and isinstance(cell, CharSpan):
                for attr in ("begin", "end", "begin_token", "end_token"):
                    if attr in dir(cell):
                        props_list.append(
                            ".property({}, {})".format(
                                _quote_str("{}.{}".format(colnames[i],
                                                          attr)),
                                _quote_str(getattr(cell, attr)))
                        )
        props_str = "".join(props_list)
        node_lines.append("""addV("token"){}.as({})""".format(
            props_str, _quote_str(index_val)))

    # Edges:
    # For each token, generate addE("head").from(token_id).to(head_id)
    edge_lines = []
    for index, value in token_features["head_token_num"].items():
        if drop_self_links and index == value:
            continue
        edge_lines.append("""addE("head").from({}).to({})""".format(
            _quote_str(index), _quote_str(value)))

    # Combine insertions into a single Gremlin statement
    result = """
    g = TinkerGraph.open().traversal()
    g.
    {}.
    {}.
    iterate()
    """.format(
        ".\n    ".join(node_lines),
        ".\n    ".join(edge_lines), )
    return textwrap.dedent(result)
