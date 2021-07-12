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
# preprocess.py
#
# utilities for preparing a corpus for inference and cleaning using other submodules
# of cleaning
#

import numpy as np
import pandas as pd

import text_extensions_for_pandas as tp
from typing import *

# Always run with the latest version of Text Extensions for Pandas
import importlib

tp = importlib.reload(tp)


def preprocess_doc_with_bert(
    document: pd.DataFrame,
    bert_model,
    tokenizer,
    token_col,
    label_col,
    keep_cols: List[str],
    iob: bool,
    iob_col=None,
):
    """
    Translates a single document in the form of a Pandas DataFrame from standard spans into BERT compatible spans,
    and calculates BERT embeddings over that text, carries over iob spans properly, as well as any additional
    information specified, on a token-by-token basis, and ensures that iob-spans are treated properly
    :param document: a PandasDataframe as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     or :func:`tp.io.conll_u_to_documents`. Must contain a column containing `span` elements and some form of
     label column, or two if IOB format is being used.
    :param bert_model: PyTorch-based BERT model from the `transformers` library.
                       A default model will be used if this is `None` or not specified

    :param tokenizer: A tokenizer that is a subclass of huggingface transformers
                       PreTrainingTokenizerFast which supports `encode_plus` with
                       return_offsets_mapping=True.
                       A default tokenizer will be used if this is `None` or not specified
    :param token_col: the column in the each of the dataframes in `document` containing
     the spans for each individual token
    :param label_col: the name of the Pandas column `document` containing the label
     over which you wish to classify (or identify incorrect elements). If using iob format
     this should be the entity type label, not the IOB label
    :param iob_col: if in iob format this must be specified. The column containing the IOB
     tag portion of the label. If using output from :func:`tp.io.conll.conll_2003_to_documents`
     this should be `'ent_col'`
    :param keep_cols: a list of names of columns that you desire to carry over to the new tokeniztion
    :param iob: if true, additional logic will be done to ensure that iob entities are treated properly
    """
    # make bert tokens
    bert_toks = tp.io.bert.make_bert_tokens(
        document.loc[0, token_col].target_text, tokenizer
    )
    raw_tok_span_array = tp.TokenSpanArray.align_to_tokens(
        bert_toks["span"], document[token_col]
    )
    raw_tok_span_ziplist = zip(
        raw_tok_span_array.begin_token, raw_tok_span_array.end_token
    )
    # add label and token if not already added
    carry_cols = keep_cols.copy()
    carry_cols = [token_col] + carry_cols if token_col not in carry_cols else carry_cols
    carry_cols = (
        [label_col] + carry_cols
        if not iob and label_col not in carry_cols
        else carry_cols
    )

    # because bert toks already has a token column of its own we need to rename this one to differentiate it.
    carry_cols_targ = [col if col != token_col else "raw_span" for col in carry_cols]
    for tok_bounds, carry_vals in zip(
        raw_tok_span_ziplist, document[carry_cols].itertuples()
    ):
        bert_toks.loc[
            tok_bounds[0] : tok_bounds[1] - 1, carry_cols_targ + ["raw_span_id"]
        ] = list(carry_vals[1:]) + [carry_vals[0]]

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
    carry_cols: List[str] = [],
    iob_col: str = "ent_iob",
    tokenizer=None,
    bert_model=None,
    token_col="span",
    show_jupyter_progress_bar=True,
    default_label_type=None,
    return_docs_as_dict=False,
    classes_list=None,
    classes_misc_val=None,
):
    """
    Take a dictionary of fold->list of documents as input, and run the full preprocessing
    sequence. This retokenizes the corpus from its original format to a BERT-compatible
    format and carries over any important information regarding it.
    It converts the label_col to a categorical dtype (allowing for iob if necessary) and
    uses the mapped outputs to create an id for each category
    Note: this function requires the `transformers` library to run.
    :param docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     or :func:`tp.io.conll_u_to_documents` All DataFrames must contain a column containing
     `span` elements and some form of label column, or two if IOB format is being used.
    :param label_col: the name of the Pandas column in each DataFrame containing the label
     over which you wish to classify (or identify incorrect elements). If using iob format
     this should be the entity type label, not the in out boundary label
    :param iob_format: boolean label indicating if the labels are in iob format or not.
    :param carry_cols: by default an empty list. Lists any columns that should be carried
     over into the output document.
    :param tokenizer: A tokenizer that is a subclass of huggingface transformers
                       PreTrainingTokenizerFast which supports `encode_plus` with
                       return_offsets_mapping=True.
                       A default tokenizer will be used if this is `None` or not specified
    :param bert_model: PyTorch-based BERT model from the `transformers` library.
                       A default model will be used if this is `None` or not specified
    :param token_col: the column in the each of the dataframes in `doc` containing the spans
     for each individual token
    :param iob_col: if in iob format this must be specified. The column containing the IOB
     tag portion of the label. If using output from :func:`tp.io.conll.conll_2003_to_documents`
     this should be `'ent_iob'`
    :param show_jupyter_progress_bar: if true, this method will use jupyter extensions to show
     a progress bar as preprocessing occurs otherwise preprocessing happens silently
    :param default_label_type: The label type that will be used if none is avaliable. If none
     is specified, this will default to 'O' if in iob format or 'X' if not. This is dependent
     on the specific classification task you are doing.

    """
    import transformers

    # input logic for default label type. Defaults to 'O' for iob, and 'X' otherwise
    if default_label_type is None:
        default_label_type = "O" if iob_format else "X"

    # initialize bert and tokenizer models if not already done
    bert_model_name = "dslim/bert-base-NER"
    if tokenizer is None:
        tokenizer = transformers.BertTokenizerFast.from_pretrained(bert_model_name)
    if bert_model is None:
        bert_model = transformers.BertModel.from_pretrained(bert_model_name)

    bert_docs_by_fold = {}
    for fold in docs.keys():
        fold_docs = docs[fold]
        if show_jupyter_progress_bar:
            print(f"preprocessing fold {fold}")
            bert_docs_by_fold[fold] = tp.jupyter.run_with_progress_bar(
                len(fold_docs),
                lambda i: preprocess_doc_with_bert(
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
                preprocess_doc_with_bert(
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
    if classes_list is None:
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
        if not return_docs_as_dict:
            corpus_df[iob_col].fillna(default_label_type, inplace=True)
            corpus_df = tp.io.conll.add_token_classes(
                corpus_df,
                classes_dtype,
                iob_col_name=iob_col,
                entity_type_col_name=label_col,
            )
        else:
            for fold in bert_docs_by_fold.keys():
                for docnum in range(len(bert_docs_by_fold[fold])):
                    bert_docs_by_fold[fold][docnum][iob_col].fillna(
                        default_label_type, inplace=True
                    )
                    bert_docs_by_fold[fold][docnum] = tp.io.conll.add_token_classes(
                        bert_docs_by_fold[fold][docnum],
                        classes_dtype,
                        iob_col_name=iob_col,
                        entity_type_col_name=label_col,
                    )

    else:
        # fill in missing or not in spec values
        corpus_df[label_col].fillna(default_label_type, inplace=True)
        if classes_misc_val is not None:
            indexes = ~corpus_df[label_col].isin(classes_list)
            corpus_df.loc[indexes, label_col] = classes_misc_val
        # create dtypes
        classes_dtype = pd.CategoricalDtype(categories=classes_list)
        classes_dict = {dtype: i for i, dtype in enumerate(classes_list)}
        # relabel
        if not return_docs_as_dict:
            corpus_df[label_col + "_id"] = corpus_df[label_col].apply(
                lambda t: classes_dict[t]
            )
            corpus_df = corpus_df.astype(
                {label_col + "_id": "int", label_col: classes_dtype}
            )
        else:
            for fold in bert_docs_by_fold.keys():
                for docnum in range(len(bert_docs_by_fold[fold])):
                    bert_docs_by_fold[fold][docnum][label_col].fillna(
                        default_label_type, inplace=True
                    )
                    bert_docs_by_fold[fold][docnum][
                        label_col + "_id"
                    ] = bert_docs_by_fold[fold][docnum][label_col].apply(
                        lambda t: classes_dict[t]
                    )
                    bert_docs_by_fold[fold][docnum] = bert_docs_by_fold[fold][
                        docnum
                    ].astype({label_col + "_id": "int", label_col: classes_dtype})

    ret = bert_docs_by_fold if return_docs_as_dict else corpus_df
    return ret, classes_dtype, classes_list, classes_dict


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


def combine_raw_spans_docs_to_match(
    raw_docs: Dict[str, List[pd.DataFrame]],
    df_to_match: pd.DataFrame,
    iob_col="ent_iob",
    token_col="span",
    label_col="ent_type",
    fold_col="fold",
    doc_col="doc_num",
):
    """
    Takes in multiple parts of a corpus and merges (i.e. train, test, validation)
    into a single DataFrame containing all the entity spans in that corpus that
    are from document-fold pairs also contianed in `df_to_match`

    This is specially intended for iob-formatted data, and converts iob labeled
    elements to spans with labels.

    :param raw_docs: Mapping from fold name ("train", "test", etc.) to
     list of per-document DataFrames as produced by :func:`tp.io.conll.conll_2003_to_documents`.
     All DataFrames must contain a tokenization in the form of a span column.
    :param df_to_match: a dataframe containing elements, and two columns, representing
     fold and document numbers respectively
    :param iob_col: the name of the column of dataframes in `raw_docs` containing
     the iob portion of the iob label. where all elements are labeled as
     either 'I','O' or 'B'
    :param token_col: the name of the column column of dataframes in `raw_docs`
     containing the surface tokens of the document
    :param label_col: the name of the column of dataframes in `raw_docs` containing
     the element type label
    :param fold_col: the name of the column in `df_to_match` containing a fold label
    :param fold_col: the name of the column in `df_to_match` containing a document number
    """

    fold_doc_pairs = (
        df_to_match[[fold_col, doc_col]]
        .drop_duplicates()
        .sort_values([fold_col, doc_col])
        .itertuples(index=False, name=None)
    )
    docs_list = []
    for fold, doc in fold_doc_pairs:
        df = tp.io.conll.iob_to_spans(
            raw_docs[fold][doc],
            iob_col_name=iob_col,
            span_col_name=token_col,
            entity_type_col_name=label_col,
        )
        df["fold"] = fold
        df[doc_col] = doc
        docs_list.append(df)
    return pd.concat(docs_list, ignore_index=True)


# alternate way to tokenize for Bert documents
def create_bert_actor_class():
    """
    Imports ray and creates a Ray actor class, BertActor.
    :returns: Ray actor class with two methods
      * __init__ which takes, a BERT model name, a token class dtype and a bool,
        compute embeddings
      * process_doc, which given a tokens dataframe runs :func:`tp.io.bert.conll_to_bert`
        on that document

    """
    import ray

    @ray.remote
    class BertActor:
        """
        Ray actor wrapper for tp.cleaning.preprocess_doc_with_bert
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

    return BertActor
