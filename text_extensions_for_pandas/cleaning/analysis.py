#
#  Copyright (c) 2021 IBM Corp.
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
###############################################################################
# util.py
#
# Cleaning utilities for finding errors in varied corpora
#

import numpy as np
import pandas as pd
import sklearn.random_projection
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import transformers

import text_extensions_for_pandas as tp

# Always run with the latest version of Text Extensions for Pandas
import importlib

tp = importlib.reload(tp)

from typing import *


def create_f1_score_report(
    predicted_features: Dict[str, pd.DataFrame],
    corpus_label_col: str,
    predicted_label_col: str,
    print_output: bool = False,
):
    """
    Takes in a set of non-IOB formatted documents such as those returned by
    `infer_and_extract_entities` as well as two column names and
    """
    if print_output:
        print(
            sklearn.metrics.classification_report(
                predicted_features[corpus_label_col],
                predicted_features[predicted_label_col],
                zero_division=0,
            )
        )
    return pd.DataFrame(
        sklearn.metrics.classification_report(
            predicted_features[corpus_label_col],
            predicted_features[predicted_label_col],
            output_dict=True,
            zero_division=0,
        )
    )


def create_f1_score_report_iob(
    predicted_ents: pd.DataFrame,
    corpus_ents: pd.DataFrame,
    span_id_col_names: List[str] = ["fold", "doc_num", "span"],
    entity_type_col_name: str = "ent_type",
    simple:bool = False
):
    """
    Calculates precision, recall and F1 scores for the given predicted elements and model
    entities. This function has two modes. In normal operation it calculates classs-wise
    precision, recall and accuacy figures, as well as global averaged metrics, and r
    eturns them as a pandas DataFrame In the 'Simple' mode, calculates micro averaged
    precision recall and F1 scorereturns them as a dictionary.
    :param predicted_ents: entities returned from the predictions of the model, in the
     form of a pandas DataFrame, with one entity per line, and some sort of 'type' column
    :param corpus_ents: the ground truth entities from the model, with one entity per line
     and some sort of entity type columns
    :param span_id_col_names: a list of column names which by themselves will be sufficent
     to uniquely identify each entity by default `['fold', 'doc_num', 'span']` to be
     compatible with outputs from `combine_raw_spans_docs`
     and `infer_and_extract_entities_iob` from this module
    :param entity_type_col_name: the name of a column in both entity DataFrames that identifies
     the type of the element.
    :param simple: by default `false`. If `false`, a full report is generated, for each entity
     type with individual precisions, recalls and F1 scores, as well as averaged metrics
     If  `true`, an dictionary with three elements `'precision'` `'recall'` and `'F1 score'`
     is returned.
    :returns: If simple is `false`, a full report is generated, for each entity
     type with individual precisions, recalls and F1 scores, as well as averaged metrics
     If simple is `true`, an dictionary with three elements `'precision'` `'recall'` and
     `'F1 score'` is returned.
    :returns:
    """
    # use an inner join to count the number of identical elts.
    inner = predicted_ents.copy().merge(
        corpus_ents, on=span_id_col_names + [entity_type_col_name], how="inner"
    )
    if simple:
        res_dict = {}
        res_dict['precision'] = inner.shape[0]/predicted_ents.shape[0]
        res_dict['recall'] = inner.shape[0]/corpus_ents.shape[0]
        res_dict['f1_score'] =( 2*res_dict['precision']*res_dict['recall']/
                                (res_dict['precision']+res_dict['recall']))
        return res_dict
    inner["true_positives"] = 1
    inner_counts = inner.groupby(entity_type_col_name).agg({"true_positives": "count"})

    pos = predicted_ents
    pos["predicted_positives"] = 1
    positive_counts = pos.groupby(entity_type_col_name).agg({"predicted_positives": "count"})

    actuals = corpus_ents
    actuals["actual_positives"] = 1
    actual_counts = actuals.groupby(entity_type_col_name).agg({"actual_positives": "count"})

    stats = pd.concat([inner_counts, positive_counts, actual_counts], axis=1)
    # add micro average
    micro = stats.sum()
    micro.name = 'Micro-avg'
    stats = stats.append(micro)
    # calc stuff
    stats['precision'] = stats.true_positives / stats.predicted_positives
    stats['recall']  = stats.true_positives / stats.actual_positives
    # macro average
    macro = stats.mean()
    macro.name = 'Macro-avg'
    stats = stats.append(macro)
    # f1 calc
    stats['f1_score'] = 2*(stats.precision*stats.recall)/(stats.precision + stats.recall)
    stats['support'] = stats['actual_positives']
    stats.loc['Micro-avg':'Macro-avg','support'] = pd.NA
    # return
    stats = stats.drop(columns =[col for col in stats.columns if 'positives' in col])
    return stats


def flag_suspicious_labels(
    predicted_features: Dict[str, pd.DataFrame],
    corpus_label_col: str,
    predicted_label_col: str,
    label_name=None,
    gold_feats: pd.DataFrame = None,
    align_over_cols: List[str] = ["fold", "doc_num", "raw_span_id"],
    keep_cols: List[str] = ["raw_span"],
):
    """
     Takes in the outputs of a number of models and and correlates the elements they
     correspond to with the respective elements in the raw corpus labels. It then
     aggregates these model results according to their values and whether or not they
     agree with the corpus.
    :returns: two pandas DataFrames:
      * `in_gold`: A DataFrame listing Elements in the corpus but with low agreement
        among the models, sorted by least agreement upwards
      * `not_in_gold`: a DataFrame listing elements that are not in the corpus labels
        but for which there is high agreement among the models of their existence
     These DataFrames have the following columns:
      * `in_gold`: boolean value of whether or not the element is in the corpus "gold standard"
      * `count`: the number of models in agreement on this datapoint
      * `models`: the list of the names of models in agreement on that datapoint as listed by
        by their names in the `predicted_features` dictionary
    """
    df_cols = align_over_cols + keep_cols
    if label_name is None:
        label_name = "class"
    # create gold features dataframe
    if gold_feats is None:
        gold_feats = predicted_features[list(predicted_features.keys())[0]]
    gold_df = gold_feats[df_cols + [corpus_label_col]].copy()
    gold_df["models"] = "GOLD"
    gold_df["in_gold"] = True
    gold_df.rename(columns={corpus_label_col: label_name}, inplace=True)
    # create list of features
    features_list = [gold_df]
    # now populate that list with all of the features from the model
    for model_name in predicted_features.keys():
        model_pred_df = predicted_features[model_name][
            df_cols + [predicted_label_col]
        ].copy()
        model_pred_df["models"] = model_name
        model_pred_df["in_gold"] = False
        model_pred_df.rename(columns={predicted_label_col: label_name}, inplace=True)
        features_list.append(model_pred_df)
    # now combine the dataframes of features and combine them with a groupby operation`
    all_features = pd.concat(features_list)
    all_features["count"] = 1
    all_features.loc[all_features.in_gold, "count"] = 0
    # create groupby aggregation dict:
    aggby = {"in_gold": "any", "count": "sum", "models": lambda x: list(x)}
    aggby.update({col: "first" for col in keep_cols})
    # now groupby
    grouped_features = (
        all_features.groupby(align_over_cols + [label_name]).agg(aggby).reset_index()
    )
    grouped_features.sort_values(
        ["count"] + align_over_cols, ascending=False, inplace=True
    )
    in_gold = grouped_features[grouped_features.in_gold].sort_values(
        "count", ascending=True, kind="mergesort"
    )
    not_in_gold = grouped_features[~grouped_features.in_gold].sort_values(
        "count", ascending=False, kind="mergesort"
    )
    return in_gold, not_in_gold
