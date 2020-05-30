###############################################################################
# util.py
#
# Utilities shared across multiple notebooks
#

import ipywidgets
from IPython.display import display
import pandas as pd
import time
import torch

import text_extensions_for_pandas as tp

from typing import *

def run_with_progress_bar(num_docs: int, fn) -> List[pd.DataFrame]:
    """
    Display a progress bar while iterating over a list of dataframes.
    
    :param num_docs: Number of documents to iterate over
    :param fn: A function that accepts a single integer argument -- let's 
     call it `i` -- and performs processing for document `i` and returns
     a `pd.DataFrame` of results 
    
    """
    _UPDATE_SEC = 0.1
    result = [] # Type: List[pd.DataFrame]
    last_update = time.time()
    progress_bar = ipywidgets.IntProgress(0, 0, num_docs,
                                          description="Starting...",
                                          layout=ipywidgets.Layout(width="100%"),
                                          style={"description_width": "12%"})
    display(progress_bar)
    for i in range(num_docs):
        result.append(fn(i))
        now = time.time()
        if i == num_docs - 1 or now - last_update >= _UPDATE_SEC:
            progress_bar.value = i + 1
            progress_bar.description = f"{i + 1}/{num_docs} docs"
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
                  token_class_dtype: pd.CategoricalDtype) -> pd.DataFrame:
    """
    :param df: One dataframe from the conll_2003_to_dataframes() function,
     representing the tokens of a single document in the original tokenization.
    :param tokenizer: BERT tokenizer instance from the `transformers` library
    :param bert: PyTorch-based BERT model from the `transformers` library
    :param token_class_dtype: Pandas categorical type for representing 
     token class labels
    
    :returns: A version of the same dataframe, but with BERT tokens, BERT
     embeddings for each token, and token class labels.
    """
    spans_df = tp.iob_to_spans(df)
    bert_toks_df = tp.make_bert_tokens(df["char_span"].values[0].target_text, 
                                       tokenizer)
    bert_token_spans = tp.TokenSpanArray.align_to_tokens(bert_toks_df["char_span"],
                                                         spans_df["token_span"])
    bert_toks_df[["ent_iob", "ent_type"]] = tp.spans_to_iob(bert_token_spans, 
                                                            spans_df["ent_type"])
    bert_toks_df = tp.add_token_classes(bert_toks_df, token_class_dtype)
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