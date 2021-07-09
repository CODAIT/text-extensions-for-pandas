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

################################################################################
# cleaning module
#
# Functions in text_extensions_for_pandas that allow for identification of 
# possibly incorrect labels, and quick training of models on bert embeddings
# of a corpus

# Expose the public APIs that users should get from importing the top-level
# library.

from text_extensions_for_pandas.cleaning import ensemble
from text_extensions_for_pandas.cleaning import analysis
from text_extensions_for_pandas.cleaning import preprocess

# import important functions from each module
from text_extensions_for_pandas.cleaning.preprocess import (
    preprocess_documents,
    combine_raw_spans_docs,
)
from text_extensions_for_pandas.cleaning.analysis import (
    flag_suspicious_labels,
    create_f1_score_report,
    create_f1_score_report_iob,
)
from text_extensions_for_pandas.cleaning.ensemble import (
    train_reduced_model,
    train_model_ensemble,
    infer_and_extract_entities_iob,
    infer_and_extract_raw_entites,
    infer_on_df,
)

__all__ = ["ensemble", "analysis", "preprocess"]
