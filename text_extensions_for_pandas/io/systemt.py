#
#  Copyright (c) 2020 IBM Corp.
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
# systemt.py
#
# I/O functions related to SystemT-format data files.

import pandas as pd

# Set to True to use sparse storage for tokens 2-n of n-token dictionary
# entries. First token is always stored dense, of course.
# Currently set to False to avoid spurious Pandas API warnings about conversion
# from sparse to dense.
# TODO: Turn this back on when Pandas fixes the issue with the warning.
_SPARSE_DICT_ENTRIES = False


def load_dict(file_name: str, tokenizer: "spacy.tokenizer.Tokenizer"):
    """
    Load a SystemT-format dictionary file. File format is one entry per line.

    Tokenizes and normalizes the dictionary entries.

    :param file_name: Path to dictionary file

    :param tokenizer: Preconfigured tokenizer object for tokenizing
    dictionary entries.  **Must be the same configuration as the tokenizer
    used on the target text!**

    :return: a `pd.DataFrame` with the normalized entries.
    """
    with open(file_name, "r") as f:
        lines = [
            line.strip() for line in f.readlines() if len(line) > 0 and line[0] != "#"
        ]

    # Tokenize with SpaCy. Produces a SpaCy document object per line.
    tokenized_entries = [tokenizer(line.lower()) for line in lines]

    # Determine the number of tokens in the longest dictionary entry.
    max_num_toks = max([len(e) for e in tokenized_entries])

    # Generate a column for each token. Go one past the max number of tokens so
    # that every dictionary entry ends up None-terminated.
    cols_dict = {}
    for i in range(max_num_toks + 1):
        # Extract token i from every entry that has a token i
        toks_list = [e[i].text if len(e) > i else None for e in tokenized_entries]
        cols_dict["toks_{}".format(i)] = (
            # Sparse storage for tokens 2 and onward
            toks_list
            if i == 0 or not _SPARSE_DICT_ENTRIES
            else pd.SparseArray(toks_list)
        )

    return pd.DataFrame(cols_dict)
