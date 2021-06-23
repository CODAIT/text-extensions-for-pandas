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
import transformers

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
    carry_cols: List[str] = [],
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

