###############################################################################
# util.py
#
# Utilities shared across multiple notebooks
#

import numpy as np
import os.path
import pandas as pd
import requests
import time
import torch
import sklearn.random_projection

import text_extensions_for_pandas as tp

# Always run with the latest version of Text Extensions for Pandas
import importlib
tp = importlib.reload(tp)

from typing import *


def train_reduced_model(x_values: np.ndarray, y_values: np.ndarray, n_components: int,
                        seed: int, max_iter: int = 10000) -> sklearn.base.BaseEstimator:
    """
    Train a reduced-quality model by putting a Gaussian random projection in
    front of the multinomial logistic regression stage of the pipeline.
    
    :param x_values: input embeddings for training set
    :param y_values: integer labels corresponding to embeddings
    :param n_components: Number of dimensions to reduce the embeddings to
    :param seed: Random seed to drive Gaussian random projection
    :param max_iter: Maximum number of iterations of L-BGFS to run. The default
     value of 10000 will achieve a tight fit but takes a while.
    
    :returns A model (Python object with a `predict()` method) fit on the
     input training data with the specified level of dimension reduction
     by random projection.
    """
    reduce_pipeline = sklearn.pipeline.Pipeline([
        ("dimred", sklearn.random_projection.GaussianRandomProjection(
            n_components=n_components,
            random_state=seed
        )),
        ("mlogreg", sklearn.linear_model.LogisticRegression(
            multi_class="multinomial",
            max_iter=max_iter
        ))
    ])
    print(f"Training model with n_components={n_components} and seed={seed}.")
    return reduce_pipeline.fit(x_values, y_values)


def predict_on_df(df: pd.DataFrame, id_to_class: Dict[int, str], predictor):
    """
    Run a trained model on a DataFrame of tokens with embeddings.

    :param df: DataFrame of tokens for a document, containing a TokenSpan column
     called "embedding" for each token.
    :param id_to_class: Mapping from class ID to class name, as returned by
     :func:`text_extensions_for_pandas.make_iob_tag_categories`
    :param predictor: Python object with a `predict` method that accepts a
     numpy array of embeddings.
    :returns: A copy of `df`, with the following additional columns:
     `predicted_id`, `predicted_class`, `predicted_iob`, and `predicted_type`
     and `predicted_class_pr`.
    """
    x_values = df["embedding"].values
    result_df = df.copy()
    result_df["predicted_id"] = predictor.predict(x_values)
    result_df["predicted_class"] = [id_to_class[i]
                                    for i in result_df["predicted_id"].values]
    iobs, types = tp.decode_class_labels(result_df["predicted_class"].values)
    result_df["predicted_iob"] = iobs
    result_df["predicted_type"] = types
    prob_values = predictor.predict_proba(x_values)
    result_df["predicted_class_pr"] = tp.TensorArray(prob_values)
    return result_df


def align_model_outputs_to_tokens(model_results: pd.DataFrame,
                                  tokens_by_doc: Dict[str, List[pd.DataFrame]]) \
    -> Dict[Tuple[str, int], pd.DataFrame]:
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
        model_results[["fold", "doc_num"]]
            .drop_duplicates()
            .to_records(index=False)
    )
    indexed_df = (
        model_results
            .set_index(["fold", "doc_num", "token_id"], verify_integrity=True)
            .sort_index()
    )
    results = {}  # Type: Dict[Tuple[str, int], pd.DataFrame]
    for collection, doc_num in all_pairs:
        doc_slice = indexed_df.loc[collection, doc_num].reset_index()
        doc_toks = tokens_by_doc[collection][doc_num][
            ["token_id", "char_span", "token_span", "ent_iob", "ent_type"]
        ].rename(columns={"id": "token_id"})
        result_df = doc_toks.copy().merge(
            doc_slice[["token_id", "predicted_iob", "predicted_type"]])
        results[(collection, doc_num)] = result_df
    return results


def analyze_model(target_df: pd.DataFrame, 
                  id_to_class: Dict[int, str],
                  predictor: Any,
                  model_tokens_by_doc: Dict[str,List[pd.DataFrame]],
                  corpus_tokens_by_doc: Dict[str,List[pd.DataFrame]] = None,
                  expand_matches: bool = True) \
        -> Dict[str, Any]:
    """
    Score a model on a target set of documents, convert the model outputs to
    spans, and compute precision and recall per document.
    
    :param target_df: Dataframe of tokens across documents with precomputed
     embeddings in a column called "embedding"
    :param id_to_class: Mapping from class ID to class name, as returned by
     :func:`text_extensions_for_pandas.make_iob_tag_categories`
    :param predictor: Trained model with a `predict()` function that accepts
     the contents of the "embedding" column of `bert_df`
    :param model_tokens_by_doc: Metadata about tokens for the tokenization 
     on which the model is based. Must include token labels as IOB2 tags.
     Indexed by fold name and index of document within fold.
    :param corpus_tokens_by_doc: Metadata about tokens for the tokenization 
     of the original corpus. Required if `expand_matches` is `True`
     Indexed by fold name and index of document within fold.
    :param expand_matches: `True` to expand model matches so that they
     align with the tokens in corpus_tokens_by_doc.
     
    :returns: A dictionary containing multiple outputs:
     * `results_by_doc`: Dictionary of dataframes of IOB2 results, indexed by
       (fold, offset into fold)
     * `actual_spans_by_doc`: Gold standard span/entity pairs, one dataframe
       per document,
     * `model_spans_by_doc`: Model output span/entity pairs, one dataframe 
       per document.
    """
    results_df = predict_on_df(target_df, id_to_class, predictor)
    
    # results_df is flat, but all other values are one dataframe per 
    # document, indexed by (fold, offset into fold)
    results_by_doc = align_model_outputs_to_tokens(results_df,
                                                      model_tokens_by_doc)
    actual_spans_by_doc = {k: tp.iob_to_spans(v) 
                           for k, v in results_by_doc.items()}
    model_spans_by_doc = {k:
        tp.iob_to_spans(v, iob_col_name = "predicted_iob",
                        entity_type_col_name = "predicted_type")
          .rename(columns={"predicted_type": "ent_type"})
        for k, v in results_by_doc.items()}
    
    if expand_matches:
        if corpus_tokens_by_doc is None:
            raise ValueError("Must supply corpus_tokens_by_doc argument "
                             "if expand_matches is True.")
        new_model_spans_by_doc = {}  # Type: Dict[Tuple[int, str], pd.DataFrame]
        for k, results_df in model_spans_by_doc.items():
            collection, doc_num = k
            tokens = corpus_tokens_by_doc[collection][doc_num]
            new_model_spans_by_doc[k] = tp.align_bert_tokens_to_corpus_tokens(results_df, tokens)
        model_spans_by_doc = new_model_spans_by_doc
    
    stats_by_doc = tp.compute_accuracy_by_document(actual_spans_by_doc,
                                                   model_spans_by_doc)
    return {
        "results_by_doc": results_by_doc,
        "actual_spans_by_doc": actual_spans_by_doc,
        "model_spans_by_doc": model_spans_by_doc,
        "stats_by_doc": stats_by_doc,
        "global_scores": tp.compute_global_accuracy(stats_by_doc)
    }


def merge_model_results(results: Dict[str, Dict[Tuple[str, int], pd.DataFrame]]) -> pd.DataFrame:
    """
    Combine the results of running :func:`util.analyze_model` across a corpus
    into a single dataframe.
    
    :param results: Mapping from model name to dictionary of dataframes,
     one dataframe per document, with the keys being tuples of (fold, index into fold)
    :returns: A single dataframe containing all results combined together, with 
     indicator variables indicating which models got which result.
    """
    model_names = list(results.keys())
    first_model_name = model_names[0]
    first_results = results[first_model_name]
    gold_standard_by_doc = first_results["actual_spans_by_doc"]
    doc_keys = list(gold_standard_by_doc.keys())
    num_docs = len(doc_keys)
    def df_for_doc(i):
        df = None
        for model_name in model_names:
            actual_spans_df = results[model_name]["actual_spans_by_doc"][doc_keys[i]]
            model_spans_df = results[model_name]["model_spans_by_doc"][doc_keys[i]]
            joined_results = pd.merge(actual_spans_df, model_spans_df, how="outer", indicator=True)
            joined_results["gold"] = joined_results["_merge"].isin(["left_only", "both"])
            joined_results[model_name] = joined_results["_merge"].isin(["right_only", "both"])
            joined_results = joined_results.drop(columns="_merge")
            if df is None:
                df = joined_results
            else:
                df = df.merge(joined_results, how="outer", 
                              on=["token_span", "ent_type", "gold"])           
        # TokenSpanArrays from different documents can't currently be stacked,
        # so convert to TokenSpan objects.
        df["token_span"] = df["token_span"].astype(object)
        df = df.fillna(False)
        vectors = df[df.columns[3:]].values
        counts = np.count_nonzero(vectors, axis=1)
        df["num_models"] = counts
        df.insert(0, "doc_offset", doc_keys[i][1])
        df.insert(0, "fold", doc_keys[i][0])
        df.insert(0, "doc_num", i)
        return df
    to_stack = tp.run_with_progress_bar(num_docs, df_for_doc)
    all_results = pd.concat(to_stack)
    return all_results


def csv_prep(counts_df: pd.DataFrame,
             counts_col_name: str):
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
    in_gold_counts = counts_df[counts_df["gold"]].sort_values(
        [counts_col_name, "fold", "doc_offset"]
    )
    in_gold_df = pd.DataFrame({
        counts_col_name: in_gold_counts[counts_col_name],
        "fold": in_gold_counts["fold"],
        "doc_offset": in_gold_counts["doc_offset"],
        "corpus_span": in_gold_counts["token_span"].astype(str),
        "corpus_ent_type": in_gold_counts["ent_type"],
        "error_type": "",
        "correct_span": "",
        "correct_ent_type": "",
        "notes": "",
        "time_started": "",
        "time_stopped": "",
        "time_elapsed": "",
    })
    
    not_in_gold_counts = counts_df[~counts_df["gold"]].sort_values(
        [counts_col_name, "fold", "doc_offset"], ascending=[False, True, True]
    )
    not_in_gold_df = pd.DataFrame({
        counts_col_name: not_in_gold_counts[counts_col_name],
        "fold": not_in_gold_counts["fold"],
        "doc_offset": not_in_gold_counts["doc_offset"],
        "model_span": not_in_gold_counts["token_span"].astype(str),
        "model_ent_type": not_in_gold_counts["ent_type"],
        "error_type": "",
        "corpus_span": "",  # Incorrect span to remove from corpus
        "corpus_ent_type": "",
        "correct_span": "",  # Correct span to add if both corpus and model are wrong
        "correct_ent_type": "",
        "notes": "",
        "time_started": "",
        "time_stopped": "",
        "time_elapsed": "",
    })
    return in_gold_df, not_in_gold_df
