
# Text Extensions for Pandas

[![Documentation Status](https://readthedocs.org/projects/text-extensions-for-pandas/badge/?version=latest)](https://text-extensions-for-pandas.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frreiss/tep-fred/branch-binder?urlpath=lab/tree/notebooks)


Natural language processing support for Pandas dataframes.

Text Extensions for Pandas turns Pandas DataFrames into a universal data
structure for representing intermediate data in all phases of your NLP
application development workflow.

## Features

### SpanArray: A Pandas extension type for *spans* of text

* Connect features with regions of a document
* Visualize the internal data of your NLP application
* Analyze the accuracy of your models
* Combine the results of multiple models

### TensorArray: A Pandas extension type for tensors

* Represent BERT embeddings in a Pandas series
* Store logits and other feature vectors in a Pandas series
* Store an entire time series in each cell of a Pandas series

### Pandas front-ends for popular NLP toolkits

* [SpaCy](https://spacy.io/)
* [Transformers](https://github.com/huggingface/transformers)
* [IBM Watson Natural Language Understanding](https://www.ibm.com/cloud/watson-natural-language-understanding)
* [IBM Watson Discovry Table Understanding](https://cloud.ibm.com/docs/discovery-data?topic=discovery-data-understanding_tables)


## Installation

This library requires Python 3.7+, Pandas, and Numpy. 

To install the latest release, just run:
```
pip install text-extensions-for-pandas
```

Depending on your use case, you may also need the following additional
packages:
* `spacy` (for SpaCy support)
* `transformers` (for 
* `ibm_watson` (for IBM Watson support)

## Installation from Source

If you'd like to try out the very latest version of our code, 
you can install directly from the head of the master branch:
```
pip install git+https://github.com/CODAIT/text-extensions-for-pandas
```

You can also directly import our package from your local copy of the 
`text_extensions_for_pandas` source tree. Just add the root of your local copy
of this repository to the front of `sys.path`.

## Documentation

For examples of how to use the library, take a look at the **example notebooks** in 
[this directory](https://github.com/CODAIT/text-extensions-for-pandas/tree/master/notebooks). You can try out these notebooks on [Binder](https://mybinder.org/) by navigating to [https://mybinder.org/v2/gh/frreiss/tep-fred/branch-binder?urlpath=lab/tree/notebooks](https://mybinder.org/v2/gh/frreiss/tep-fred/branch-binder?urlpath=lab/tree/notebooks)

To run the notebooks on your local machine, follow the following steps:

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Check out a copy of this repository.
1. Use the script `env.sh` to set up an Anaconda environment for running the code in this repository.
1. Type `jupyter lab` from the root of your local source tree to start a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) environment.
1. Navigate to the `notebooks` directory and choose any of the notebooks there

API documentation can be found at [https://text-extensions-for-pandas.readthedocs.io/en/latest/](https://text-extensions-for-pandas.readthedocs.io/en/latest/)


## Contents of this repository

* **`text_extensions_for_pandas`**: Source code for the `text_extensions_for_pandas` module.
* **env.sh**: Script to create a conda environment `pd` capable of running the notebooks and test cases in this project
* **generate_docs.sh**: Script to build the [API documentation]((https://readthedocs.org/projects/text-extensions-for-pandas/)
* **api_docs**: Configuration files for `generate_docs.sh`
* **binder**: Configuration files for [running notebooks on Binder](https://mybinder.org/v2/gh/frreiss/tep-fred/branch-binder?urlpath=lab/tree/notebooks)
* **config**: Configuration files for `env.sh`.
* **docs**: Project web site
* **notebooks**: example notebooks
* **resources**: various input files used by our example notebooks 
* **test_data**: data files for regression tests. The tests themselves are
  located adjacent to the library code files.
* **tutorials**: Detailed tutorials on using Text Extensions for Pandas to
  cover complex end-to-end NLP use cases (work in progress).



## Contributing

This project is an IBM open source project. We are developing the code in the open under the [Apache License](https://github.com/CODAIT/text-extensions-for-pandas/blob/master/LICENSE), and we welcome contributions from both inside and outside IBM. 

To contribute, just open a Github issue or submit a pull request. Be sure to include a copy of the [Developer's Certificate of Origin 1.1](https://elinux.org/Developer_Certificate_Of_Origin) along with your pull request.


## Building and Running Tests

Before building the code in this repository, we recommend that you use the 
provided script `env.sh` to set up a consistent build environment:
```
$ ./env.sh --env_name myenv
$ conda activate myenv
```
(replace `myenv` with your choice of environment name).

To run tests, navigate to the root of your local copy and run:
```
pytest
```

To build pip and source code packages:

```
python setup.py sdist bdist_wheel
```

(outputs go into `./dist`).

To build API documentation, run:

```
./generate_docs.sh
```




