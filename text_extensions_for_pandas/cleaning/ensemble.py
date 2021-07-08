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
# ensemble.py
#
# Cleaning utilities training and running ensembles of reduced models on BERT embeddings,
# for use with analysis.py to identify potentially incorrect labels
#

import numpy as np
import pandas as pd
import text_extensions_for_pandas as tp

# Always run with the latest version of Text Extensions for Pandas
import importlib

tp = importlib.reload(tp)

from typing import *


def train_reduced_model(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_components: int,
    seed: int,
    max_iter: int = 10000,
):
    """
    Train a reduced-quality model by putting a Gaussian random projection in
    front of the multinomial logistic regression stage of the pipeline.
    Requires `sklearn` and `ray` packages to run

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
    import sklearn.pipeline
    import sklearn.random_projection
    import sklearn.linear_model

    reduce_pipeline = sklearn.pipeline.Pipeline(
        [
            (
                "dimred",
                sklearn.random_projection.GaussianRandomProjection(
                    n_components=n_components, random_state=seed
                ),
            ),
            (
                "mlogreg",
                sklearn.linear_model.LogisticRegression(
                    multi_class="multinomial", max_iter=max_iter
                ),
            ),
        ]
    )
    print(f"Training model with n_components={n_components} and seed={seed}.")
    return reduce_pipeline.fit(x_values, y_values)


def train_model_ensemble(
    training_data: pd.DataFrame,
    labels_col: str,
    x_feats_col: str = "embedding",
    model_sizes=None,
    model_seeds=None,
    max_iters=10000,
):
    """
    Train an ensemble of reduced-quality models by putting a Gaussian
    random projection in front of the multinomial logistic regression
     stage of the pipelines for a set of models
     requires `sklearn` and `ray` packages to run

    two lists are given of model sizes and seeds, and the power set
     of the two is the complete set ofparameters used to train the models
    Uses Ray to speed up model training.
    :param training_data: a dataframe containing the bert embeddings and
     labels for the models to train on.
    :param labels_col: the name of the column containing the labels for the model
     to train on
    :param x_feats_col: the name of the column containing the BERT embeddings
     for each token, off which the model trains
    :param model_sizes: the number of components that the gaussian random progression
     reduces the BERT embedding to.
    :param model_seeds: seeds for the random initialization of the model.
    :param max_iters: the upper bound on the number of iterations to allow
     the models to train. 100 is fast and 10,000 typically means full convergence
    :returns: A dictionary mapping model names to models (Python object with a
     `predict()` method) fit on the input training data with the specified
     level of dimension reduction by random projection.
    """

    import ray  # TODO: put a note about this in the docstring
    import sklearn.pipeline

    # input logic
    if model_sizes is None:
        model_sizes = [32, 64, 128, 256]
    model_sizes.reverse()
    if model_seeds is None:
        model_seeds = [1, 2, 3]
    model_params = {
        f"{size}_{seed}": (size, seed) for size in model_sizes for seed in model_seeds
    }
    # training data sets
    X_train = training_data[x_feats_col].values
    Y_train = training_data[labels_col]
    # run ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    # wrapper func for ray reduced model training
    @ray.remote
    def train_reduced_model_task(
        x_values: np.ndarray,
        y_values: np.ndarray,
        n_components: int,
        seed: int,
        max_iter: int = max_iters,
    ) -> sklearn.base.BaseEstimator:
        return train_reduced_model(x_values, y_values, n_components, seed, max_iter)

    # setup plasma
    X_id = ray.put(X_train.to_numpy())
    Y_id = ray.put(Y_train.to_numpy())
    # run training
    futures = [
        train_reduced_model_task.remote(
            X_id, Y_id, components, seed, max_iter=max_iters
        )
        for components, seed in model_params.values()
    ]
    results = ray.get(futures)
    # Clean up items we've added to Plasma and shut down ray
    del X_id
    del Y_id
    ray.shutdown()
    models = {name: model for name, model in zip(model_params.keys(), results)}
    return models


def infer_on_df(
    df: pd.DataFrame, id_to_class_dict, predictor, iob=False, embeddings_col="embedding"
):
    """
    Takes a dataframe containing bert embeddings and a model trained on bert embeddings,
    and runs inference on the dataframe. if IOB is specified, predicted id and type are
    broken out from the raw probabilities given.
    :param df: the document on which to perform inference; of the form output by  the
     `preprocess_documents` method of this module, and containing BERT embeddings,
     references to fold and document numbers, as well as some column containing unique
     identifiers for the raw tokenization of the document (i.e. `'raw_token_id'` field in
     output DataFrames from `preprocess_documents`)
    :param id_to_class_dict:  Mapping from class ID to class name, as returned by
      :func:`text_extensions_for_pandas.make_iob_tag_categories`
    :param predictor: Python object with a `predict` method that accepts a
     numpy array of embeddings.
    :param iob: a boolean value, when set to true, additional logic for iob-formatted
     classes is activated
    :param embeddings_col: the column in `df` that contains BERT embeddings for that document
    :returns: a Pandas DataFrame, mirroring df, and conaining three extra columns:
        *  `'predicted_id'` with the id as predicted by the model of the categorical element
        *  `'predicted_class'` containing the predicted categorical value corresponding to
            predicted_id
        *  `'raw_output'` a TensorArray containing the raw output vectors from the model
    """
    result_df = df.copy()
    raw_outputs = tp.TensorArray(predictor.predict_proba(result_df[embeddings_col]))
    result_df["predicted_id"] = np.argmax(raw_outputs, axis=1)
    result_df["predicted_class"] = result_df["predicted_id"].apply(
        lambda p_id: id_to_class_dict[p_id]
    )
    if iob:
        iobs, types = tp.io.conll.decode_class_labels(
            result_df["predicted_class"].values
        )
        result_df["predicted_iob"] = iobs
        result_df["predicted_type"] = types
    result_df["raw_output"] = raw_outputs

    return result_df


def infer_and_extract_raw_entites(
    doc: pd.DataFrame,
    id_to_class_dict,
    predictor,
    raw_span_id_col="raw_span_id",
    fold_col="fold",
    doc_col="doc_num",
    agg_func=None,
    keep_cols: List[str] = None,
):
    """
    Takes a dataframe containing bert embeddings and a model trained on bert embeddings, and
    runs inference on the dataframe. Then using references to the original spans, reconstucts
    the predicted value of each token of the original tokenization.
    :param doc: the document on which to perform inference; of the form output by  the
     `preprocess_documents` method of this module, and containing BERT embeddings, references to
     fold and document numbers, as well as some column containing unique identifiers for the raw
     tokenization of the document
    :param id_to_class_dict:  Mapping from class ID to class name, as returned by
      :func:`text_extensions_for_pandas.make_iob_tag_categories`
    :param predictor: Python object with a `predict` method that accepts a
     numpy array of embeddings.
    :param fold_col: the name of the column of `doc` containing the fold of each token
    :param doc_col: the name of the column of `doc` containing the document number of each token
    :param raw_span_id_col: the name of the column of `doc` containing some identifier of the raw
      token that each bert token came from.
    :param agg_func: if specified, a function that takes in a series of tensorArrays and returns a
      Pandas-compatible type; used to aggregate the predictions of multiple subtokens when
      multiple subtokens all describe the same original token.
    :param keep_cols: any column that you wish to be carried over to the output dataframe, by default
      the column 'raw_span' is the only column to be carried over, if it exists.
    :returns: a Pandas DataFrame containing a set of entities aligned with the orignal
     tokenization of the document, containing the following columns:
        * `'predicted_id'` the id number of the predicted element
        *  `'raw_output'` a vector of prediction 'probabilities' from the model. If the
           entity span covers multiple tokens, it is aggregated using agg_func
        *  `'predicted_class'` the class of the entity, matching predicted_id, and converted
            using `id_to_class_dict`
        * any columns specified in `keep_cols`
    """
    if agg_func is None:

        def agg_func(series: pd.Series):
            # util function for predicting the probabilities of each class when multiple sub-tokens are combined.
            # this method assumes independence between subtoken classes and calculates the probabilities of
            # all subtokens being the same class, then re-normalizes so the vector components sum to one again
            vec = series.to_numpy().prod(axis=0)
            if (
                np.sum(vec) == 0
            ):  # if we underflow, (only happens in rare cases) log everything and continue
                mat = np.log2(series.to_numpy())
                vec = mat.sum(axis=0)
                vec -= np.logaddexp2.reduce(vec)
                return np.exp2(vec)

            return tp.TensorArray(vec / np.sum(vec))

    # build aggregation fields
    keep_cols = (
        keep_cols
        if keep_cols is not None
        else [
            "fold",
            "doc_num",
            "token_id",
            "raw_span",
        ]
    )
    sort_cols = [
        col for col in [fold_col, doc_col, raw_span_id_col] if col in doc.columns
    ]
    keep_cols = [
        c for c in keep_cols if c in doc.columns and c not in sort_cols
    ]  # filter out cols not in df
    aggby = {k: "first" for k in keep_cols}
    aggby["raw_output"] = agg_func
    df = doc[["embedding"] + keep_cols + sort_cols].copy()
    # first, run inference
    df.loc[:, "raw_output"] = tp.TensorArray(predictor.predict_proba(df["embedding"]))
    # group by original tag
    groupby = df.groupby(sort_cols)
    results_df = groupby.agg(aggby).reset_index().sort_values(sort_cols)
    # repeat translation
    results_df["predicted_id"] = results_df.raw_output.apply(
        lambda s: np.array(s).argmax()
    )

    results_df["predicted_class"] = results_df["predicted_id"].apply(
        lambda p_id: id_to_class_dict[p_id]
    )
    return results_df


def extract_entities_iob(
    predicted_df: pd.DataFrame,
    raw_docs: Dict[str, List[pd.DataFrame]],
    span_col="span",
    fold_col="fold",
    doc_col="doc_num",
    iob_col="predicted_iob",
    entity_type_col="predicted_type",
    raw_docs_span_col_name="span",
):
    """
    Takes a dataframe containing bert embeddings and a model trained on bert embeddings, and
    runs inference on the dataframe. Then using a reference to the surface form of the document
    converts the iob-type entities into entity spans, that align with the original tokenization
    of the document.
    **This method is specifically for IOB-formatted labels**
    :param predicted_df: Document in BERT tokenization with inferred element types and iob tags.
      as well as token span, doc number, and fold information for each token. This can be the
      output from :func: `pd.cleaning.ensemble.infer_on_df`.
    :param raw_docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     thes DataFrames must contain the original tokenization in the form of text-extensions spans
    :param span_col: the name of the column of `doc` containing the surface tokens as spans
    :param fold_col: the name of the column of `doc` containing the fold of each token
    :param doc_col: the name of the column of `doc` containing the document number of each token
    :param raw_docs_span_col_name: the name of the column of the documents in `raw_docs` containing
     the tokens of those documents as spans.
    :param iob_col: the column containing the predicted iob values from the model
    :param entity_type_col: the column containing the predicted element types from the model
    :returns: a Pandas DataFrame containing the extracted entities from the predicted iob
     tags, with each iob-labelled element as its own line, and with the following columns:
      * `'span'` containing the spans of the entities flagged by the model
      * `'ent_type'` with the predicted type of the flagged entity
      as well as two columns containing the fold and doc numbers for each element, using
      the same names specified in `fold_col` and `doc_col` respecively
    """

    # create predicted spans using inference
    pred_dfs = []
    for fold, doc_num in (
        predicted_df[[fold_col, doc_col]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        pred_doc = predicted_df[
            (predicted_df[fold_col] == fold) & (predicted_df[doc_col] == doc_num)
        ].reset_index()
        pred_spans = tp.io.conll.iob_to_spans(
            pred_doc,
            iob_col_name=iob_col,
            span_col_name=span_col,
            entity_type_col_name=entity_type_col,
        )
        pred_spans.rename(columns={entity_type_col: "ent_type"}, inplace=True)
        pred_aligned_doc = tp.io.bert.align_bert_tokens_to_corpus_tokens(
            pred_spans, raw_docs[fold][doc_num].rename({raw_docs_span_col_name: "span"})
        )
        pred_aligned_doc.rename(columns={"ent_type": entity_type_col})
        pred_aligned_doc[[fold_col, doc_col]] = [fold, doc_num]
        pred_dfs.append(pred_aligned_doc)
    result_df = pd.concat(pred_dfs)
    return result_df


def infer_and_extract_entities_iob(
    doc: pd.DataFrame,
    raw_docs: Dict[str, List[pd.DataFrame]],
    id_to_class_dict,
    predictor,
    span_col="span",
    fold_col="fold",
    doc_col="doc_num",
    raw_docs_span_col_name="span",
    predict_on_col="embedding",
):
    """
    Takes a dataframe containing bert embeddings and a model trained on bert embeddings, and
    runs inference on the dataframe. Then using a reference to the surface form of the document
    converts the iob-type entities into entity spans, that align with the original tokenization
    of the document.
    **This method is designed specifically for IOB-formatted labels**
    :param doc: the document on which to perform inference; of the form output by  the
     `preprocess_documents` method of this module, and containing BERT embeddings, references to
     fold and document numbers, as well as some column representing the tokens of the doucment
    :param raw_docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     thes DataFrames must contain the original tokenization in the form of text-extensions spans
    :param id_to_class_dict:  Mapping from class ID to class name, as returned by
      :func:`text_extensions_for_pandas.make_iob_tag_categories`
    :param predictor: Python object with a `predict` method that accepts a
     numpy array of embeddings.
    :param span_col: the name of the column of `doc` containing the surface tokens as spans
    :param fold_col: the name of the column of `doc` containing the fold of each token
    :param doc_col: the name of the column of `doc` containing the document number of each token
    :param predict_on_col: the name of the column of `doc` containing the BERT embedding of that token
    :param raw_docs_span_col_name: the name of the column of the documents in `raw_docs` containing
     the tokens of those documents as spans.
    :returns: a Pandas DataFrame containing the predicted entities from the model,
      converted from iob format with each element as its own line,
      and with the following columns:
      * `'span'` containing the spans of the entities flagged by the model
      * `'ent_type'` with the predicted type of the flagged entity
      as well as two columns containing the fold and doc numbers for each element, using
      the same names specified in `fold_col` and `doc_col` respecively

    """

    df = doc.copy()
    # construct raw text from dataframe

    # first, run inference
    predicted_df = infer_on_df(
        df, id_to_class_dict, predictor, embeddings_col=predict_on_col, iob=True
    )
    return extract_entities_iob(
        predicted_df,
        raw_docs,
        span_col=span_col,
        fold_col=fold_col,
        doc_col=doc_col,
        raw_docs_span_col_name=raw_docs_span_col_name,
    )
