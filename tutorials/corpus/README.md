# Tutorial: Identifying Incorrect labels in the CoNLL-2003 Corpus

This directory holds the Jupyter notebooks that were used for the experiments
in the CoNLL-2020 paper, ["Identifying Incorrect Labels in the CoNLL-2003 Corpus"](https://www.aclweb.org/anthology/2020.conll-1.16/).

Currently these notebooks consist primarily of code. We intend to add detailed explanations 
of each step as we convert this code into an in-depth tutorial on using Text Extensions
for Pandas for corpus-level analysis of model outputs and gold-standard labels.
If you would like to be updated on our progress adding explanatory text to these notebooks, 
subscribe to the updates for the associated [GitHub issue](https://github.com/CODAIT/text-extensions-for-pandas/issues/148).

Summary of the contents this directory:

* [CoNLL_2.ipynb](CoNLL_2.ipynb): Aggregates the outputs of the 16 models submitted to the CoNLL-2003 competition. Compares these outputs to the corpus's gold-standard labels and identifies areas where there is a strong consensus between model outputs coupled with a disagreement with the corpus labels. Writes out CSV files containing ranked lists of potentially-incorrect labels.

* [CoNLL_3.ipynb](CoNLL_3.ipynb): Computes BERT embeddings at every token position in the CoNLL-2003 corus. Uses the "train" fold of the corpus to train an ensemble of 17 different models over the embeddings. Evaluates the models over all three folds of the corpus, then uses the same aggregation techniques used in `CoNLL_2.ipynb` to flag potentially-incorrect labels. Writes out CSV files containing ranked lists of potentially-incorrect labels.

* [CoNLL_4.ipynb](CoNLL_4.ipynb): Repeats the model training process from `CoNLL_3.ipynb`, but performs a 10-fold cross-validation. This process involves training a total of 170 models -- 10 groups of 17. Evaluates each group of models over the holdout set from the associated fold of the cross-validation, then aggregates together these outputs and uses the same techniques used in `CoNLL_2.ipynb` to flag potentially-incorrect labels. Writes out CSV files containing ranked lists of potentially-incorrect labels.

* [CoNLL_View_Doc.ipynb](CoNLL_View_Doc.ipynb): Notebook for displaying the gold standard labels for one document at a time. Our human annotators used this notebook to review the labels that the other notebooks in this directory flagged.

* [outputs](outputs): Output directory for the notebooks. This is where all output files, as well as the cached copy of the CoNLL-2003 corpus, are written.

You should be able to run all of these notebooks on any machine with at least 4GB of memory. Follow the instructions in the main `README.md` file for this project to set up a JupyterLab environment that can run these notebooks.

If you're new to the Text Extensions for Pandas library, we recommend that you start
by reading through the notebook [`Analyze_Model_Outputs.ipynb`](https://github.com/CODAIT/text-extensions-for-pandas/blob/master/notebooks/Analyze_Model_Outputs.ipynb), which explains the 
portions of the library that we use in the notebooks in this directory.
