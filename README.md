
# Text Extensions for Pandas

[![Documentation Status](https://readthedocs.org/projects/text-extensions-for-pandas/badge/?version=latest)](https://text-extensions-for-pandas.readthedocs.io/en/latest/?badge=latest)

Natural language processing support for Pandas dataframes.

Text Extensions for Pandas turns Pandas DataFrames into a universal data
structure for representing intermediate data in all phases of your NLP
application development workflow.

## Features:

* `SpanArray`: A Pandas extension type for representing *spans* of text 
* `TensorAray`: A Pandas extension type for n-dimensional array values   
* Integrations with popular NLP toolkits
* Functions for common NLP tasks over Pandas DataFrames


## Installation

This library requires Python 3.7+, Pandas, and Numpy. 

To install the latest release, just run:
```
pip install text-extensions-for-pandas
```

If you'd like to try out the very latest version of our code, 
you can install directly from the head of the master branch:
```
pip install git+https://github.com/CODAIT/text-extensions-for-pandas
```

You can also directly import our package from your local copy of the 
`text_extensions_for_pandas` source tree. Just add the root of your local copy
of this repository to the front of `sys.path`.

## Documentation

For examples of how to use the library, take a look at the notebooks in 
[this directory](https://github.com/CODAIT/text-extensions-for-pandas/tree/master/notebooks).

API documentation can be found at [https://readthedocs.org/projects/text-extensions-for-pandas/](https://readthedocs.org/projects/text-extensions-for-pandas/).


## Contents of this repository

* **`text_extensions_for_pandas`**: Source code for the `text_extensions_for_pandas` module.
* **env.sh**: Script to create an conda environment `pd` capable of running the notebooks and test cases in this project
* **generate_docs.sh**: Script to build the [API documentation]((https://readthedocs.org/projects/text-extensions-for-pandas/)
* **api_docs**: Configuration files for `generate_docs.sh`
* **config**: Configuration files for `env.sh`.
* **docs**: Project web site
* **notebooks**: example notebooks
* **resources**: various input files used by our example notebooks 
* **test_data**: data files for regression tests. The tests themselves are
  located adjacent to the library code files.
* **tutorials**: Detailed tutorials on using Text Extensions for Pandas to
  cover complex end-to-end NLP use cases (work in progress).


## Instructions to run a demo notebook
1. Check out a copy of this repository
1. (optional) Use the script `env.sh` to set up an Anaconda environment for running the code in this repository.
1. Type `jupyter lab` from the root of your local source tree to start a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) environment.
1. Navigate to the `notebooks` directory and choose any of the notebooks there


## Contributing

This project is an IBM open source project. We are developing the code in the open under the [Apache License](https://github.com/CODAIT/text-extensions-for-pandas/blob/master/LICENSE), and we welcome contributions from both inside and outside IBM. 

To contribute, just open a Github issue or submit a pull request. Be sure to include a copy of the [Developer's Certificate of Origin 1.1](https://elinux.org/Developer_Certificate_Of_Origin) along with your pull request.

## Running Tests

To run regression tests:
1. (optional) Use the script `env.sh` to set up an Anaconda environment
1. Run `python -m unittest discover` from the root of your local copy

## Building and Running Tests

Before building the code in this repository, we recommend that you use the 
provided script `env.sh` to set up a consistent build environment:
```
$ ./env.sh myenv
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




