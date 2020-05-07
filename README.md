# Text Extensions for Pandas
Natural language processing support for Pandas dataframes.

**This project is under development.** Releases are not yet available.

## Purpose of this Project

Natural language processing (NLP) applications tend to consist of multiple components tied together in a complex pipeline. These components can range from deep parsers and machine learning models to lookup tables and business rules. All of them work by creating and manipulating data structures that represent data about the target text --- things like tokens, entities, parse trees, and so on.

Libraries for common NLP tasks tend to implement their own custom data structures. They also implement basic low-level operations like filtering and pattern matching over these data structures. For example, `nltk` represents named entities as a list of Python objects:

```python
>>> entities = nltk.chunk.ne_chunk(tagged)
>>> entities
Tree('S', [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'),
           ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN'),
       Tree('PERSON', [('Arthur', 'NNP')]),
           ('did', 'VBD'), ("n't", 'RB'), ('feel', 'VB'),
           ('very', 'RB'), ('good', 'JJ'), ('.', '.')])
```

...while SpaCy represents named entities with the an `Iterable` of `Span` objects:

```python
>>> doc = nlp("At eight o'clock on Thursday morning, Arthur didn't feel very good.")
>>> ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
>>> ents
[("eight o'clock", 3, 16, 'TIME'), ('Thursday', 20, 28, 'DATE'), ('morning', 29, 36, 'TIME'), ('Arthur', 38, 44, 'PERSON')]
```

...or an `Iterable` of `Token` objects with tags:

```python
>>> doc = nlp("At eight o'clock on Thursday morning, Arthur didn't feel very good.")
>>> token_info = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
>>> token_info
[('At', 'O', ''), ('eight', 'B', 'TIME'), ("o'clock", 'I', 'TIME'), ('on', 'O', ''), ('Thursday', 'B', 'DATE'), ('morning', 'B', 'TIME'), (',', 'O', ''), ('Arthur', 'B', 'PERSON'), ('did', 'O', ''), ("n't", 'O', ''), ('feel', 'O', ''), ('very', 'O', ''), ('good', 'O', ''), ('.', 'O', '')]
```

...and IBM Watson Natural Language Understanding represents named entities as an array of JSON records:

```JSON
{
  "entities": [
    {
      "type": "Person",
      "text": "Arthur",
      "count": 1,
      "confidence": 0.986158
    }
  ]
}
```

This duplication leads to a great deal of redundant work when building NLP applications.  Developers need to understand and remember how every component represents every type of data. They need to write code to convert among different representations, and they and need to implement common operations like pattern matching multiple times for different, equivalent data structures.

It is our belief that, with a few targeted improvements, we can make [Pandas](https://pandas.pydata.org/) dataframes into a universal representation for all the data that flows through NLP applications. Such a universal data structure would eliminate redundancy and make application code simpler, faster, and easier to debug.

This project aims to create the extensions that will turn Pandas into this universal data structure. In particular, we plan to add three categories of extension:

* **New Pandas series types to cover spans and tensors.** These types of data are very important for NLP applications but are cumbersome to represent with "out-of-the-box" Pandas. The new [extensions API](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html) that Pandas released in 2019 makes it possible to create performant extension types. We will use this API to add three new series types: CharSpan, TokenSpan (span with token offsets), and Tensor. 
* **An implementation of spanner algebra over Pandas dataframes.** The core operations of the [Document Spanners](https://researcher.watson.ibm.com/researcher/files/us-fagin/jacm15.pdf)formalism represent tasks that occur repeatedly in NLP applications. Many of these core operations are already present in Pandas. We will create high-performance implementations of the remaining operations over Pandas dataframes. This work will build directly on our Pandas extension types for representing spans.
* **An implementation of the Gremlin graph query language over Pandas dataframes.** As one of the most widely used graph query languages, [Gremlin](https://tinkerpop.apache.org/gremlin.html) is a natural choice for NLP tasks that involve parse trees and knowledge graphs. There are many graph database systems that support Gremlin, including [Apache TinkerPop](https://tinkerpop.apache.org/gremlin.html), [JanusGraph](https://docs.janusgraph.org/basics/gremlin/), [Neo4J](https://github.com/neo4j-contrib/gremlin-plugin), [Amazon Neptune](https://docs.aws.amazon.com/neptune/latest/userguide/access-graph-gremlin.html), [Azure CosmosDB](https://docs.microsoft.com/en-us/azure/cosmos-db/graph-modeling), and [IBM Db2 Graph](https://pdfs.semanticscholar.org/acb7/f2cea33f79a212b26eaa6e38dca5c7867786.pdf). However, using Gremlin in Python programs is difficult today, as the Python support of existing Gremlin providers is generally weak. We will create an embedded Gremlin engine that operates directly over Pandas dataframes. This embedded engine will give NLP developers the power of a graph query language without having to manage an external graph database.

## Getting Started

### Contents of this repository

* **`text_extensions_for_pandas`**: Source code for the `text_extensions_for_pandas` module.
* **notebooks**: demo notebooks
* **resources**: various input files used by the demo notebooks 
* **env.sh**: Script to create an conda environment `pd` capable of running the notebooks in this directory

### Instructions to run a demo notebook
1. Check out a copy of this repository
1. (optional) Use the script `env.sh` to set up an Anaconda environment for running the code in this repository.
1. Type `jupyter lab` from the root of your local source tree to start a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) environment.
1. Navigate to the example notebook `notebooks/Person.ipynb`

### Installation instructions

We have not yet posted a release of this project, but you can install by
building a `pip` package or by directly importing the contents of the
`text_extensions_for_pandas` source tree.

To build a pip package from your local copy:
1. (optional) Activate the `pd` environment that `env.sh` creates
1. `python3 setup.py sdist bdist_wheel`
1. The package's `.whl` file will appear under the `dist` directory.

To build and install a pip package from the head of the master branch:
```
pip install git+https://github.com/CODAIT/text-extensions-for-pandas
```

To directly import the contents of the `text_extensions_for_pandas` source tree 
as a Python package:
1. Add the root directory of your local copy of this repository to the
   front of 
```python
import text_extensions_for_pandas as tp
```

## Contributing

This project is an IBM open source project. We are developing the code in the open under the [Apache License](https://github.com/CODAIT/text-extensions-for-pandas/blob/master/LICENSE), and we welcome contributions from both inside and outside IBM. 

To contribute, just open a Github issue or submit a pull request. Be sure to include a copy of the [Developer's Certificate of Origin 1.1](https://elinux.org/Developer_Certificate_Of_Origin) along with your pull request.

## Running Tests

To run regression tests:
1. (optional) Use the script `env.sh` to set up an Anaconda environment
1. Run `python -m unittest discover` from the root of your local copy

