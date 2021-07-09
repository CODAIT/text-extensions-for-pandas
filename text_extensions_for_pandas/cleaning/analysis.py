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
# analysis.py
#
# Cleaning utilities for analyzing model outputs and flagging potentially incorrect
# labels

import numpy as np
import pandas as pd
from typing import *

import text_extensions_for_pandas as tp

# Always run with the latest version of Text Extensions for Pandas
import importlib

tp = importlib.reload(tp)



def create_f1_score_report(
    predicted_features: Dict[str, pd.DataFrame],
    corpus_label_col: str,
    predicted_label_col: str,
):
    """
    Takes in a set of non-IOB formatted documents such as those returned by
    `infer_and_extract_entities` as well as two column names and returns a
    Pandas DataFrame with the per-category precision, recall and F1 scores.
    Requires sklearn.metrics.
    if desired, a printout of the dataframe is printed as output.
    :param predicted_features: a DataFrame containing predicted outputs from
      the model, as well as the corpus labels for those same elements
    :param corpus_label_col: the name of the `predicted_features` column that
      contains the corpus labels for the entitity types
    :param predicted_label_col: the name of the `predicted_features` column that
      contains the predicted labels for the entitity types
    :returns: A dataframe containing four columns: `'precision'`, `'recall;`
      `'f1-score'` and `'support'` with one row for each entity type, as well as
      three additional rows containing accuracy, micro averaged and macro averaged
      scores.
    """
    import sklearn.metrics

    df = pd.DataFrame(
        sklearn.metrics.classification_report(
            predicted_features[corpus_label_col],
            predicted_features[predicted_label_col],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    return df


def create_f1_score_report_iob(
    predicted_ents: pd.DataFrame,
    corpus_ents: pd.DataFrame,
    span_id_col_names: List[str] = ["fold", "doc_num", "span"],
    entity_type_col_name: str = "ent_type",
    simple: bool = False,
):
    """
    Calculates precision, recall and F1 scores for the given predicted elements and model
    entities. This function has two modes. In normal operation it calculates classs-wise
    precision, recall and accuacy figures, as well as global averaged metrics, and r
    eturns them as a Pandas DataFrame In the 'Simple' mode, calculates micro averaged
    precision recall and F1 scorereturns them as a dictionary.
    :param predicted_ents: entities returned from the predictions of the model, in the
     form of a Pandas DataFrame, with one entity per line, and some sort of 'type' column
     with a name specified in `entity_type_col_name`
    :param corpus_ents: the ground truth entities from the model, with one entity per line
     and some sort of entity type columns
    :param span_id_col_names: a list of column names which by themselves will be sufficent
     to uniquely identify each entity by default `['fold', 'doc_num', 'span']` to be
     compatible with outputs from `combine_raw_spans_docs`
     and `infer_and_extract_entities_iob` from this module
    :param entity_type_col_name: the name of a column in both entity DataFrames that identifies
     the type of the element.
    :param simple: by default `false`. If `false`, a Pandas DataFrame is returned
      with four columns: `'precision'`, `'recall;`,`'f1-score'` and `'support'`
      with one row for each entity type, as well as two additional rows
      micro averaged and macro averaged scores.
      If  `true`, an dictionary with three elements `'precision'` `'recall'` and `'f1-score'`
      is returned.
    :returns: If `simple` is `false`, a Pandas DataFrame is returned
      with four columns: `'precision'`, `'recall;`,`'f1-score'` and `'support'`
      with one row for each entity type, as well as two additional rows
      micro averaged and macro averaged scores.
      If `simple` is `true`, an dictionary with three elements `'precision'` `'recall'` and `'f1-score'`
      is returned.
    """
    # use an inner join to count the number of identical elts.
    # TODO: create a regression test to check zero-predicted-ents Behaviour
    if predicted_ents.shape[0] == 0:
        if simple:
            return {"precision": 0, "recall": 0, "f1-score": 0}
        else:
            zero_by_rows = {name: 0 for name in ["Macro-avg", "Micro-avg"]}
            zeros_df = pd.DataFrame(
                {
                    name: zero_by_rows
                    for name in ["precision", "recall", "f1-score", "support"]
                }
            )
            return zeros_df
    inner = predicted_ents.copy().merge(
        corpus_ents, on=span_id_col_names + [entity_type_col_name], how="inner"
    )
    if simple:
        res_dict = {}
        res_dict["precision"] = inner.shape[0] / predicted_ents.shape[0]
        res_dict["recall"] = inner.shape[0] / corpus_ents.shape[0]
        res_dict["f1-score"] = (
            2
            * res_dict["precision"]
            * res_dict["recall"]
            / (res_dict["precision"] + res_dict["recall"])
        )
        return res_dict
    inner["true_positives"] = 1
    inner_counts = inner.groupby(entity_type_col_name).agg({"true_positives": "count"})

    pos = predicted_ents
    pos["predicted_positives"] = 1
    positive_counts = pos.groupby(entity_type_col_name).agg(
        {"predicted_positives": "count"}
    )

    actuals = corpus_ents
    actuals["actual_positives"] = 1
    actual_counts = actuals.groupby(entity_type_col_name).agg(
        {"actual_positives": "count"}
    )

    stats = pd.concat([inner_counts, positive_counts, actual_counts], axis=1)
    # add micro average
    micro = stats.sum()
    micro.name = "Micro-avg"
    stats = stats.append(micro)
    # calc stuff
    stats["precision"] = stats.true_positives / stats.predicted_positives
    stats["recall"] = stats.true_positives / stats.actual_positives
    # macro average
    macro = stats.mean()
    macro.name = "Macro-avg"
    stats = stats.append(macro)
    # f1 calc
    stats["f1-score"] = (
        2 * (stats.precision * stats.recall) / (stats.precision + stats.recall)
    )
    stats["support"] = stats["actual_positives"]
    # return
    stats.loc["Macro-avg", "support"] = stats.loc["Micro-avg", "support"]
    stats = stats.drop(columns=[col for col in stats.columns if "positives" in col])
    return stats


def create_f1_report_ensemble_iob(
    predicted_ents_by_model: Dict[str, pd.DataFrame],
    corpus_ents: pd.DataFrame,
    span_id_col_names: List[str] = ["fold", "doc_num", "span"],
    entity_type_col_name: str = "ent_type",
):
    """
    Given an ensemble of model predictions (in the form of entities) and ground truth
    labels creates a precision-recall-f1_score report for each model, and returns the
    output as a Pandas DataFrame. The outputs are of the same form as the simple output
    from :func:`create_f1_score_report_iob`
    :param predicted_ents_by_model: a dictionary from model name (or other unique
     identifier) to outputs as produced by
     :func:`cleaning.ensenble.infer_and_extract_entities_iob` or analagous.
     Must have one of each column in `span_id_col_names` and some entity type column
    :param corpus_ents: the entities given in the corpus. in the form of a Pandas DataFrame
     Must have one of each column name in `span_id_col_names` and `entity_type_col_name`
     Can be produced by :func: `cleaning.preprocess.combine_raw_spans_docs`
    :param span_id_col_names: a list column names in all input dataFrames by which each
     span may be uniquely identified. By default, `["fold", "doc_num", "span"]`
    :param entity_type_col_name: the name of the column in the input DataFrames containing
     the entity type labels for each entity.
    :returns: a Pandas DataFrame with indices of the model names, and columns
     `'precision'` `'recall'` and `'f1-score'`
    """
    reports = {
        name: create_f1_score_report_iob(
            df,
            corpus_ents,
            span_id_col_names=span_id_col_names,
            entity_type_col_name=entity_type_col_name,
            simple=True,
        )
        for name, df in predicted_ents_by_model.items()
    }
    return pd.DataFrame.from_dict(reports).transpose()


def flag_suspicious_labels(
    predicted_features: Dict[str, pd.DataFrame],
    corpus_label_col: str,
    predicted_label_col: str,
    label_name=None,
    gold_feats: pd.DataFrame = None,
    align_over_cols: List[str] = ["fold", "doc_num", "raw_span_id"],
    keep_cols: List[str] = ["raw_span"],
    count_name: str = "count",
    split_doc: bool = True,
):
    """
     Takes in the outputs of a number of models and and correlates the elements they
     correspond to with the respective elements in the raw corpus labels. It then
     aggregates these model results according to their values and whether or not they
     agree with the corpus.
    :returns: two Pandas DataFrames:
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
    if count_name != "count":
        grouped_features.rename(columns={"count", count_name}, inplace=True)
    # return
    if not split_doc:
        return grouped_features
    else:
        in_gold = grouped_features[grouped_features.in_gold].sort_values(
            "count", ascending=True, kind="mergesort"
        )
        not_in_gold = grouped_features[~grouped_features.in_gold].sort_values(
            "count", ascending=False, kind="mergesort"
        )
        return in_gold, not_in_gold


# used for document-by-document seperation and alignment
def align_model_outputs_to_tokens(
    model_results: pd.DataFrame, tokens_by_doc: Dict[str, List[pd.DataFrame]]
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Join the results of running a model on an entire corpus back with multiple
    DataFrames of the token features for the individual documents.

    :param model_results: DataFrame containing results of prediction over a
     collection of documents. Must have the fields:
     * `fold`: What fold of the original collection each document came from
     * `doc_num`: Index of the document within the fold
     * `token_id`: Token offset of the token
     * `predicted_iob`/`predicted_type`: Model outputs
     Usually this DataFrame is the result of running :func:`predict_on_df()`
    :param tokens_by_doc: One DataFrame of tokens and labels per document,
     indexed by fold and document number (which must align with the values
     in the "fold" and "doc_num" columns of `model_results`).
     These DataFrames must contain columns `ent_iob` and `ent_type` that
     correspond to the `predicted_iob` and `predicted_type` values in
     `model_results`

    :returns: A dictionary that maps (collection, offset into collection)
     to DataFrame of results for that document
    """
    all_pairs = (
        model_results[["fold", "doc_num"]].drop_duplicates().to_records(index=False)
    )
    indexed_df = model_results.set_index(
        ["fold", "doc_num", "token_id"], verify_integrity=True
    ).sort_index()
    results = {}  # Type: Dict[Tuple[str, int], pd.DataFrame]
    for collection, doc_num in all_pairs:
        doc_slice = indexed_df.loc[collection, doc_num].reset_index()
        doc_toks = tokens_by_doc[collection][doc_num][
            ["token_id", "span", "ent_iob", "ent_type"]
        ].rename(columns={"id": "token_id"})
        result_df = doc_toks.copy().merge(
            doc_slice[["token_id", "predicted_iob", "predicted_type"]]
        )
        results[(collection, doc_num)] = result_df
    return results


def csv_prep(
    counts_df: pd.DataFrame,
    counts_col_name: str,
    gold_col_name: str = "in_gold",
    fold_col_name: str = "fold",
    doc_col_name="doc_num",
    span_col_name: str = "span",
    ent_type_col_name: str = "ent_type",
):
    """
    Reformat a dataframe of results to prepare for writing out
    CSV files for hand-labeling.

    :param counts_df: Dataframe of entities with counts of who
     found each entity
    :param counts_col_name: Name of column in `counts_df` with the
     number of teams/models/etc. who found each entity

    Returns two dataframes. The first dataframe contains entities that ARE
    in the gold standard. The second dataframe contains formatted results
    for the entities that are NOT in the gold standard but are in at least
    one model's output.
    """
    # Reformat the results to prepare for hand-labeling a spreadsheet
    in_gold_counts = counts_df[counts_df[gold_col_name]].sort_values(
        [counts_col_name, fold_col_name, doc_col_name]
    )
    in_gold_df = pd.DataFrame(
        {
            counts_col_name: in_gold_counts[counts_col_name],
            "fold": in_gold_counts[fold_col_name],
            "doc_offset": in_gold_counts[doc_col_name],
            "corpus_span": in_gold_counts[span_col_name].astype(str),
            "corpus_ent_type": in_gold_counts[ent_type_col_name],
            "error_type": "",
            "correct_span": "",
            "correct_ent_type": "",
            "notes": "",
            "time_started": "",
            "time_stopped": "",
            "time_elapsed": "",
        }
    )

    not_in_gold_counts = counts_df[~counts_df[gold_col_name]].sort_values(
        [counts_col_name, fold_col_name, doc_col_name], ascending=[False, True, True]
    )
    not_in_gold_df = pd.DataFrame(
        {
            counts_col_name: not_in_gold_counts[counts_col_name],
            "fold": not_in_gold_counts[fold_col_name],
            "doc_offset": not_in_gold_counts[doc_col_name],
            "model_span": not_in_gold_counts[span_col_name].astype(str),
            "model_ent_type": not_in_gold_counts[ent_type_col_name],
            "error_type": "",
            "corpus_span": "",  # Incorrect span to remove from corpus
            "corpus_ent_type": "",
            "correct_span": "",  # Correct span to add if both corpus and model are wrong
            "correct_ent_type": "",
            "notes": "",
            "time_started": "",
            "time_stopped": "",
            "time_elapsed": "",
        }
    )
    return in_gold_df, not_in_gold_df
