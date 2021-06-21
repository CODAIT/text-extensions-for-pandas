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
import ray
import transformers

import text_extensions_for_pandas as tp

# Always run with the latest version of Text Extensions for Pandas
import importlib

tp = importlib.reload(tp)

from typing import *

# Define a Ray actor to compute embeddings.
@ray.remote
class BertActor:
    """
    Ray actor wrapper for tp.io.bert.conll_to_bert()
    """

    def __init__(
        self,
        bert_model_name: str,
        token_class_dtype: Any,
        compute_embeddings: bool = True,
    ):
        import transformers as trf

        self._tokenizer = trf.BertTokenizerFast.from_pretrained(
            bert_model_name,
            add_special_tokens=True,
        )
        self._tokenizer.deprecation_warnings[
            "sequence-length-is-longer-than-the-specified-maximum"
        ] = True
        self._bert = trf.BertModel.from_pretrained(bert_model_name)
        self._token_class_dtype = token_class_dtype
        self._compute_embeddings = compute_embeddings

    def process_doc(self, tokens_df):
        return tp.io.bert.conll_to_bert(
            tokens_df,
            self._tokenizer,
            self._bert,
            self._token_class_dtype,
            compute_embeddings=self._compute_embeddings,
        )


def train_reduced_model(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_components: int,
    seed: int,
    max_iter: int = 10000,
) -> sklearn.base.BaseEstimator:
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


def _preprocess_with_bert(
    document: pd.DataFrame,
    bert_model,
    tokenizer,
    token_col,
    label_col,
    carry_cols_in: List[str],
    iob: bool,
    iob_col=None,
):
    """
    Translates the document from standard spans into BERT compatible spans, and calculates BERT embeddings,
    carries over iob spans properly, as well as any additional information specified, on a span-by-span basis,
    and ensures that iob-spans are treated properly
    """
    # make bert tokens
    bert_toks = tp.io.bert.make_bert_tokens(
        document.loc[0, token_col].target_text, tokenizer
    )
    raw_tok_spans = (
        tp.TokenSpanArray.align_to_tokens(bert_toks["span"], document[token_col])
        .as_frame()
        .drop(columns=["begin", "end", "covered_text"])
    )
    # add label and token if not already added
    carry_cols = carry_cols_in.copy()
    carry_cols = [token_col] + carry_cols if token_col not in carry_cols else carry_cols
    carry_cols = (
        [label_col] + carry_cols
        if not iob and label_col not in carry_cols
        else carry_cols
    )

    # because bert toks already has a token column of its own we need to rename this one to differentiate it.
    carry_cols_targ = [col if col != token_col else "raw_span" for col in carry_cols]
    raw_tok_spans[carry_cols_targ] = document[carry_cols]
    for i, b_tok, e_tok, *carrys in raw_tok_spans.itertuples():
        bert_toks.loc[b_tok : e_tok - 1, carry_cols_targ + ["raw_span_id"]] = carrys + [
            i
        ]

    # if we're in iob, we use inbuilt functions to handle beginnings specially
    if iob:
        ent_spans_raw = tp.io.conll.iob_to_spans(
            document,
            iob_col_name=iob_col,
            span_col_name=token_col,
            entity_type_col_name=label_col,
        )
        ent_spans_aligned = tp.TokenSpanArray.align_to_tokens(
            bert_toks["span"], ent_spans_raw["span"]
        )
        bert_toks[[iob_col, label_col]] = tp.io.conll.spans_to_iob(
            ent_spans_aligned, span_ent_types=ent_spans_raw["ent_type"]
        )
    return tp.io.bert.add_embeddings(bert_toks, bert_model)


def preprocess_documents(
    docs: Dict[str, List[pd.DataFrame]],
    label_col: str,
    iob_format: bool,
    carry_cols: List[str]=[],
    iob_col: str = None,
    tokenizer=None,
    bert_model=None,
    token_col="span",
    show_jupyter_progress_bar=True,
    default_label_type=None,
):
    """
    Take a dictionary of fold->list of documents as input, and run the full preprocessing
    sequence. This retokenizes the corpus from its original format to a BERT-compatible 
    format and carries over any important information regarding it.
    It converts the label_col to a categorical dtype (allowing for iob if necessary) and 
    uses the mapped outputs to create an id for each category
    :param Docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     or `tp.io.conll_u_to_documents` All DataFrames must contain a column containing 
     `span` elements and some form of label column, or two if IOB format is being used.
    :param label_col: the name of the pandas column in each DataFrame containing the label 
     over which you wish to classify (or identify incorrect elements). If using iob format 
     this should be the entity type label, not the in out boundary label
    :param iob_format: boolean label indicating if the labels are in iob format or not.
    :param carry_cols: by default an empty list. Lists any columns that should be carried
     over into the output document.
    :param: tokenizer: A tokenizer that is a subclass of huggingface transformers
                       PreTrainingTokenizerFast which supports `encode_plus` with
                       return_offsets_mapping=True.
                       A default tokenizer will be used if this is `None` or not specified
    :param bert: PyTorch-based BERT model from the `transformers` library.
                       A default model will be used if this is `None` or not specified
    :param span_col

    """
    # input logic for default label type. Defaults to 'O' for iob, and 'X' otherwise
    if default_label_type is None:
        default_label_type = "O" if iob_format else "X"

    # initialize bert and tokenizer models if not already done
    bert_model_name = "dslim/bert-base-NER"
    if tokenizer is None: 
        tokenizer =  transformers.BertTokenizerFast.from_pretrained(bert_model_name)
    if bert_model is None:
        bert_model = transformers.BertModel.from_pretrained(bert_model_name)

    bert_docs_by_fold = {}
    for fold in docs.keys():
        fold_docs = docs[fold]
        if show_jupyter_progress_bar:
            print(f"preprocessing fold {fold}")
            bert_docs_by_fold[fold] = tp.jupyter.run_with_progress_bar(
                len(fold_docs),
                lambda i: _preprocess_with_bert(
                    fold_docs[i],
                    bert_model,
                    tokenizer,
                    token_col,
                    label_col,
                    carry_cols,
                    iob_format,
                    iob_col=iob_col,
                ),
            )
        else:
            bert_docs_by_fold[fold] = [
                _preprocess_with_bert(
                    doc,
                    bert_model,
                    tokenizer,
                    token_col,
                    label_col,
                    carry_cols,
                    iob_format,
                    iob_col=iob_col,
                )
                for doc in fold_docs
            ]

    # Now combine docs, into a single large dataframe
    corpus_df = tp.io.conll.combine_folds(bert_docs_by_fold)
    # and finally, translate the label column
    classes_list = list(
        corpus_df.loc[corpus_df[label_col].notnull(), label_col].unique()
    )

    if default_label_type not in classes_list and not iob_format:
        classes_list.append(default_label_type)
    # create classes (if iob or not)
    if iob_format:
        # create dtypes
        if "O" in classes_list:
            classes_list.remove("O")  # this gets added by the iob func

        # if iob, use a special function to add in the I/B/O tags
        classes_dtype, classes_list, classes_dict = tp.io.conll.make_iob_tag_categories(
            classes_list
        )
        # relabel
        corpus_df[iob_col].fillna("O", inplace=True)
        corpus_df = tp.io.conll.add_token_classes(
            corpus_df,
            classes_dtype,
            iob_col_name=iob_col,
            entity_type_col_name=label_col,
        )
    else:
        corpus_df[label_col].fillna(default_label_type, inplace=True)
        # create dtypes
        classes_dtype = pd.CategoricalDtype(categories=classes_list)
        classes_dict = {dtype: i for i, dtype in enumerate(classes_list)}
        # relabel
        corpus_df[label_col + "_id"] = corpus_df[label_col].apply(
            lambda t: classes_dict[t]
        )
        corpus_df = corpus_df.astype(
            {label_col + "_id": "int", label_col: classes_dtype}
        )

    return corpus_df, classes_dtype, classes_list, classes_dict


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
      pandas-compatible type; used to aggregate the predictions of multiple subtokens when
      multiple subtokens all describe the same original token.
    :param keep_cols: any column that you wish to be carried over to the output dataframe, by default
      the column 'raw_span' is the only column to be carried over, if it exists.
    """
    if agg_func is None:

        def agg_func(series: pd.Series):
            # util function for predicting the probabilities of each class when multiple sub-tokens are combined.
            # this method assumes independence between subtoken classes and calculates the probabilities of
            # all subtokens being the same class, then re-normalizes so the vector components sum to one again
            vec = series.to_numpy().prod(axis=0)
            if np.sum(vec) ==0: # if we underflow, (only happens in rare cases) log everything and continue 
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
    :param token_col: the name of the column of `doc` containing the surface tokens as spans
    :param fold_col: the name of the column of `doc` containing the fold of each token
    :param doc_col: the name of the column of `doc` containing the document number of each token
    :param embedding: the name of the column of `doc` containing the BERT embedding of that token
    :param raw_docs_span_col_name: the name of the column of the documents in `raw_docs` containing
     the tokens of those documents as spans.
    """

    df = doc.copy()
    # construct raw text from dataframe

    # first, run inference
    predicted_df = infer_on_df(
        df, id_to_class_dict, predictor, embeddings_col=predict_on_col, iob=True
    )
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
            iob_col_name="predicted_iob",
            span_col_name=span_col,
            entity_type_col_name="predicted_type",
        )
        pred_spans.rename(columns={"predicted_type": "ent_type"}, inplace=True)
        pred_aligned_doc = tp.io.bert.align_bert_tokens_to_corpus_tokens(
            pred_spans, raw_docs[fold][doc_num].rename({raw_docs_span_col_name: "span"})
        )
        pred_aligned_doc[[fold_col, doc_col]] = [fold, doc_num]
        pred_dfs.append(pred_aligned_doc)
    result_df = pd.concat(pred_dfs)
    return result_df


def combine_raw_spans_docs(
    docs: Dict[str, List[pd.DataFrame]], iob_col, token_col, label_col
):
    """
    Takes in multiple parts of a corpus and merges (i.e. train, test, validation)
    into a single DataFrame containing all the entity spans in that corpus

    This is specially intended for iob-formatted data, and converts iob labeled
    elements to spans with labels.

    :param docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     All DataFrames must contain a tokenization in the form of a span column.
    :param iob_col: the name of the column of the datframe containing the iob portion of
      the iob label. where all elements are labeled as either 'I','O' or 'B'
    :param token_col: the name of the column of the datframe containing the surface tokens
      of the document
    :param label_col: the name of the column of the datframe containing the element type label
    """
    docs_dict = {}
    for fold in docs.keys():
        docs_dict[fold] = [
            tp.io.conll.iob_to_spans(
                document,
                iob_col_name=iob_col,
                span_col_name=token_col,
                entity_type_col_name=label_col,
            )
            for document in docs[fold]
        ]
    return tp.io.conll.combine_folds(docs_dict)


def flag_suspicious_labels(
    predicted_features: Dict[str, pd.DataFrame],
    corpus_label_col,
    predicted_label_col,
    label_name=None,
    gold_feats: pd.DataFrame = None,
    align_over_cols: List[str] = ["fold", "doc_num", "raw_span_id"],
    keep_cols: List[str] = ['raw_span'],
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
    gold_df.rename(columns={corpus_label_col:label_name}, inplace=True)
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
    # now combine the dataframes of features and combine them with a groupby operation
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
