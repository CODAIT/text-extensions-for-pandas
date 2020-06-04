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
# watson.py
#
# I/O functions related to Watson Natural Language Processing on the IBM Cloud.

import pandas as pd
import pyarrow as pa


def watson_nlp_parse_response(response):

    def flatten_dict(d, parent_name=None):
        flattened = {}
        for k, v in d.items():
            if isinstance(v, dict):
                child = flatten_dict(v, k)
                flattened.update(child)
            elif isinstance(v, list) and isinstance(v[0], dict):
                if parent_name is not None:
                    k = parent_name + "." + k
                elements = [flatten_dict(i, k) for i in v]
                flattened[k] = elements
            else:
                if parent_name is not None:
                    k = parent_name + "." + k
                flattened[k] = v
        return flattened

    dfs = {}

    # Create the entities DataFrame
    entities = response.get("entities", [])
    entities = [flatten_dict(i) for i in entities]
    entities_df = pd.DataFrame.from_records(entities)
    dfs["entities"] = entities_df

    # Create the keywords DataFrame
    keywords = response.get("keywords", [])
    keywords = [flatten_dict(i) for i in keywords]
    keywords_df = pd.DataFrame.from_records(keywords)
    dfs["keywords"] = keywords_df

    # Create the relations DataFrame
    relations = response.get("relations", [])
    relations = [flatten_dict(i) for i in relations]
    relations_df = pd.DataFrame.from_records(relations)
    dfs["relations"] = relations_df

    return dfs


def watson_nlp_parse_response_arrow(response):

    def flatten_struct(struct_array, parent_name=None):
        arrays = struct_array.flatten()
        fields = [f for f in struct_array.type]
        for array, field in zip(arrays, fields):
            name = field.name if parent_name is None else parent_name + "." + field.name
            if pa.types.is_struct(array.type):
                for child_array, child_name in flatten_struct(array, name):
                    yield child_array, child_name
            elif pa.types.is_list(array.type) and pa.types.is_struct(array.type.value_type):
                struct = array.flatten()
                for child_array, child_name in flatten_struct(struct, name):
                    list_array = pa.ListArray.from_arrays(array.offsets, child_array)
                    yield list_array, child_name
            else:
                yield array, name

    def make_dataframe(records):
        if len(records) > 0:
            arr = pa.array(records)
            assert pa.types.is_struct(arr.type)
            arrays, names = zip(*flatten_struct(arr))
            table = pa.Table.from_arrays(arrays, names)
            return table.to_pandas()
        else:
            # TODO: fill in with expected schema
            return pd.DataFrame.empty

    dfs = {}

    # Create the entities DataFrame
    entities = response.get("entities", [])
    dfs["entities"] = make_dataframe(entities)

    # Create the keywords DataFrame
    keywords = response.get("keywords", [])
    dfs["keywords"] = make_dataframe(keywords)

    # Create the relations DataFrame
    relations = response.get("relations", [])
    dfs["relations"] = make_dataframe(relations)

    # Create the semantic roles DataFrame
    semantic_roles = response.get("semantic_roles", [])
    dfs["semantic_roles"] = make_dataframe(semantic_roles)

    # Create the syntax DataFrame
    syntax = response.get("syntax", {})
    paragraph = syntax.get("paragraphs", [])
    dfs["syntax.paragraph"] = make_dataframe(paragraph)
    sentence = syntax.get("sentences", [])
    dfs["syntax.sentence"] = make_dataframe(sentence)
    tokens = syntax.get("tokens", [])
    dfs["syntax.tokens"] = make_dataframe(tokens)

    return dfs
