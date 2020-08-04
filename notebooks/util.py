###############################################################################
# util.py
#
# Utilities shared across multiple notebooks
#

import ipywidgets
from IPython.display import display
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

def run_with_progress_bar(num_items: int, fn: Callable, item_type: str = "doc") \
        -> List[pd.DataFrame]:
    """
    Display a progress bar while iterating over a list of dataframes.
    
    :param num_items: Number of items to iterate over
    :param fn: A function that accepts a single integer argument -- let's 
     call it `i` -- and performs processing for document `i` and returns
     a `pd.DataFrame` of results 
    :param item_type: Human-readable name for the items that the calling
     code is iterating over
    
    """
    _UPDATE_SEC = 0.1
    result = [] # Type: List[pd.DataFrame]
    last_update = time.time()
    progress_bar = ipywidgets.IntProgress(0, 0, num_items,
                                          description="Starting...",
                                          layout=ipywidgets.Layout(width="100%"),
                                          style={"description_width": "12%"})
    display(progress_bar)
    for i in range(num_items):
        result.append(fn(i))
        now = time.time()
        if i == num_items - 1 or now - last_update >= _UPDATE_SEC:
            progress_bar.value = i + 1
            progress_bar.description = f"{i + 1}/{num_items} {item_type}s"
            last_update = now
    progress_bar.bar_style = "success"
    return result


def add_embeddings(df: pd.DataFrame, bert: Any) -> pd.DataFrame:
    """
    Add BERT embeddings to a dataframe of BERT tokens.
    
    :param df: Dataframe containing BERT tokens. Must contain a column
     "input_id" containing token IDs.
    :param bert: PyTorch-based BERT model from the `transformers` library
    :returns: A copy of `df` with a new column, "embedding" containing
     BERT embeddings as a `TensorArray`.
    """
    _OVERLAP = 32
    _NON_OVERLAP = 64
    flat_input_ids = df["input_id"].values
    windows = tp.seq_to_windows(flat_input_ids, _OVERLAP, _NON_OVERLAP)
    bert_result = bert(
        input_ids=torch.tensor(windows["input_ids"]), 
        attention_mask=torch.tensor(windows["attention_masks"]))
    hidden_states = tp.windows_to_seq(flat_input_ids, 
                                      bert_result[0].detach().numpy(),
                                      _OVERLAP, _NON_OVERLAP)
    embeddings = tp.TensorArray(hidden_states)
    ret = df.copy()
    ret["embedding"] = embeddings
    return ret

def conll_to_bert(df: pd.DataFrame, tokenizer, bert, 
                  token_class_dtype: pd.CategoricalDtype,
                  compute_embeddings: bool = True) -> pd.DataFrame:
    """
    :param df: One dataframe from the conll_2003_to_dataframes() function,
     representing the tokens of a single document in the original tokenization.
    :param tokenizer: BERT tokenizer instance from the `transformers` library
    :param bert: PyTorch-based BERT model from the `transformers` library
    :param token_class_dtype: Pandas categorical type for representing 
     token class labels
    :param compute_embeddings: True to generate BERT embeddings at each token
     positiona and add a column "embedding" to the returned dataframe with
     the embeddings
     
    :returns: A version of the same dataframe, but with BERT tokens, BERT
     embeddings for each token (if `compute_embeddings` is `True`), 
     and token class labels.
    """
    spans_df = tp.iob_to_spans(df)
    bert_toks_df = tp.make_bert_tokens(df["char_span"].values[0].target_text, 
                                       tokenizer)
    bert_token_spans = tp.TokenSpanArray.align_to_tokens(bert_toks_df["char_span"],
                                                         spans_df["token_span"])
    bert_toks_df[["ent_iob", "ent_type"]] = tp.spans_to_iob(bert_token_spans, 
                                                            spans_df["ent_type"])
    bert_toks_df = tp.add_token_classes(bert_toks_df, token_class_dtype)
    if compute_embeddings:
        bert_toks_df = add_embeddings(bert_toks_df, bert)
    return bert_toks_df

def combine_folds(fold_to_docs: Dict[str, List[pd.DataFrame]]):
    """
    Merge together multiple parts of a corpus (i.e. train, test, validation)
    into a single dataframe of all tokens in the corpus.
    
    :param fold_to_docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document dataframes as produced by :func:`util.conll_to_bert`
    
    :returns: corpus wide DataFrame with some additional leading columns `fold`
     and `doc_num` to tell what fold and document number within the fold each 
     row of the dataframe comes from.
    """
    def prep_for_stacking(fold_name: str, doc_num: int, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": fold_name,
            "doc_num": doc_num,
            "token_id": df["id"],
            "ent_iob": df["ent_iob"],
            "ent_type": df["ent_type"],
            "token_class": df["token_class"],
            "token_class_id": df["token_class_id"],
            "embedding": df["embedding"]
        })
    
    to_stack = []  # Type: List[pd.DataFrame]
    for fold_name, docs_in_fold in fold_to_docs.items():
        to_stack.extend([
            prep_for_stacking(fold_name, i, docs_in_fold[i])
            for i in range(len(docs_in_fold))])

    return pd.concat(to_stack).reset_index(drop=True)


def predict_on_df(df: pd.DataFrame, id_to_class: Dict[int, str], predictor):
    """
    Run a trained model on a dataframe of tokens with embeddings.
    
    :param df: DataFrame of tokens for a document, containing the a column
     "embedding" for each token.
    :param id_to_class: Mapping from class ID to class name, as returned by
     :func:`text_extensions_for_pandas.make_iob_tag_categories`
    :param predictor: Python object with a `predict` method that accepts a
     numpy array of embeddings.
    :returns: A copy of `df`, with the following additional columns:
     `predicted_id`, `predicted_class`, `predicted_iob`, and `predicted_type`.
    """
    id_to_class = df["token_class"].values.categories.values

    X = df["embedding"].values
    result_df = df.copy()
    result_df["predicted_id"] = predictor.predict(X)
    result_df["predicted_class"] = [id_to_class[i] for i in result_df["predicted_id"].values]
    iobs, types = tp.decode_class_labels(result_df["predicted_class"].values)
    result_df["predicted_iob"] = iobs
    result_df["predicted_type"] = types
    return result_df


def align_model_outputs_to_tokens(model_results: pd.DataFrame, 
                                  tokens_by_doc: Dict[str,List[pd.DataFrame]]) \
        -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Join the results of running a model on an entire set of documents back with
    dataframes of the token features for the individual documents.
    
    :param model_results: Dataframe containing results of prediction over a 
     collection of documents. Should have the fields:
     * "fold": What fold of the original collection each document came from
     * "doc_num": Index of the document within the fold
     * "token_id": Token offset of the token
     * "predicted_iob"/"predicted_type": Model outputs
    :param tokens_by_doc: One dataframe of tokens and labels per document, 
     indexed by fold and document number (which must align with the values
     in the "fold" and "doc_num" columns of `model_results`)
    
    :returns: A dictionary that maps (collection, offset into collection) 
     to dataframe of results for that document
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
            ["id", "char_span", "token_span", "ent_iob", "ent_type"]
        ].rename(columns={"id": "token_id"})
        result_df = doc_toks.copy().merge(
            doc_slice[["token_id", "predicted_iob", "predicted_type"]])
        results[(collection, doc_num)] = result_df
    return results


def train_reduced_model(X: np.ndarray, Y: np.ndarray, n_components: int, 
                        seed: int) -> sklearn.base.BaseEstimator:
    """
    Train a reduced-quality model by putting a Gaussian random projection in
    front of the multinomial logistic regression stage of the pipeline.
    
    :param X: input embeddings for training set
    :param Y: integer labels corresponding to embeddings
    :param n_components: Number of dimensions to reduce the embeddings to
    :param seed: Random seed to drive Gaussian random projection
    
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
            max_iter=10000
        ))
    ])
    print(f"Training model with n_components={n_components} and seed={seed}.")
    return reduce_pipeline.fit(X, Y)


def make_stats_df(gold_dfs: Dict[Tuple[str, int], pd.DataFrame],
                  output_dfs: Dict[Tuple[str, int], pd.DataFrame]) \
        -> pd.DataFrame:
    """
    Compute precision and recall statistics by document.
    
    :param gold_dfs: Gold-standard span/entity type pairs, as dictionary 
     of dataframes, one dataframe per document, indexed by tuples of 
     (collection name, offset into collection)
    :param output_dfs: Model outputs, in the same format as `gold_dfs`
     (i.e. exactly the same column names)
    """
    # Note that it's important for all of these lists to be in the same
    # order; hence these expressions all iterate over gold_dfs.keys()
    num_true_positives = [
        len(gold_dfs[k].merge(output_dfs[k]).index) 
        for k in gold_dfs.keys()]
    num_extracted = [len(output_dfs[k].index) for k in gold_dfs.keys()]
    num_entities = [len(gold_dfs[k].index)for k in gold_dfs.keys()]
    collection_name = [t[0] for t in gold_dfs.keys()]
    doc_num = [t[1] for t in gold_dfs.keys()]

    stats_by_doc = pd.DataFrame({
        "fold": collection_name,
        "doc_num": doc_num,
        "num_true_positives": num_true_positives,
        "num_extracted": num_extracted,
        "num_entities": num_entities
    })
    stats_by_doc["precision"] = stats_by_doc["num_true_positives"] / stats_by_doc["num_extracted"]
    stats_by_doc["recall"] = stats_by_doc["num_true_positives"] / stats_by_doc["num_entities"]
    stats_by_doc["F1"] = (
        2.0 * (stats_by_doc["precision"] * stats_by_doc["recall"]) 
        / (stats_by_doc["precision"] + stats_by_doc["recall"]))
    return stats_by_doc


def compute_global_scores(stats_by_doc: pd.DataFrame):
    """
    Compute collection-wide precision, recall, and F1 score from the 
    output of :func:`make_stats_df`.
    
    :param stats_by_doc: Output of :func:`make_stats_df`
    :returns: A Python dictionary of collection-level statistics about
     result quality.
    """
    num_true_positives = stats_by_doc["num_true_positives"].sum()
    num_entities = stats_by_doc["num_entities"].sum()
    num_extracted = stats_by_doc["num_extracted"].sum()

    precision = num_true_positives / num_extracted
    recall = num_true_positives / num_entities
    F1 = 2.0 * (precision * recall) / (precision + recall)
    return {
        "num_true_positives": num_true_positives,
        "num_entities": num_entities,
        "num_extracted": num_extracted,
        "precision": precision,
        "recall": recall,
        "F1": F1
    }


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
     of the original corpus. Required if `realign_to_tokens` is `True`
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
                             "if realign_to_tokens is True.")
        new_model_spans_by_doc = {}  # Type: Dict[Tuple[int, str], pd.DataFrame]
        for k, results_df in model_spans_by_doc.items():
            collection, doc_num = k
            tokens = corpus_tokens_by_doc[collection][doc_num]
            new_model_spans_by_doc[k] = realign_to_tokens(results_df, tokens)
        model_spans_by_doc = new_model_spans_by_doc
    
    stats_by_doc = make_stats_df(actual_spans_by_doc, model_spans_by_doc)
    return {
        "results_by_doc": results_by_doc,
        "actual_spans_by_doc": actual_spans_by_doc,
        "model_spans_by_doc": model_spans_by_doc,
        "stats_by_doc": stats_by_doc,
        "global_scores": compute_global_scores(stats_by_doc)
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
    to_stack = run_with_progress_bar(num_docs, df_for_doc)
    all_results = pd.concat(to_stack)
    return all_results


def realign_to_tokens(spans_df: pd.DataFrame, corpus_toks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand entity matches from a BERT-based model so that they align
    with the corpus's original tokenization.
    
    :param spans_df: DataFrame of extracted entities. Must contain two
     columns: "token_span" and "ent_type". Other columns ignored.
    :param corpus_toks_df: DataFrame of the corpus's original tokenization,
     one row per token.
     Must contain a column "char_span" with character-based spans of
     the tokens.
     
    :returns: A new dataframe with the schema ["token_span", "ent_type"],
     where the "token_span" column contains token-based spans based off
     the *corpus* tokenization in `corpus_toks_df["char_span"]`.
    """
    if len(spans_df.index) == 0:
        return spans_df.copy()
    overlaps_df = (
        tp
        .overlap_join(spans_df["token_span"], corpus_toks_df["char_span"],
                     "token_span", "corpus_token")
        .merge(spans_df)
    )
    agg_df = (
        overlaps_df
        .groupby("token_span")
        .aggregate({"corpus_token": "sum", "ent_type": "first"})
        .reset_index()
    )
    cons_df = (
        tp.consolidate(agg_df, "corpus_token")
        [["corpus_token", "ent_type"]]
        .rename(columns={"corpus_token": "token_span"})
    )
    cons_df["token_span"] = tp.TokenSpanArray.align_to_tokens(
            corpus_toks_df["char_span"], cons_df["token_span"])
    return cons_df

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
