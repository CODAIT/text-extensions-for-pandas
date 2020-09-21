<!-- Package description for PyPI -->

# Text Extensions for Pandas

Natural language processing support for Pandas DataFrames.

Text Extensions for Pandas adds [extension types](https://pandas.pydata.org/docs/development/extending.html) to Pandas DataFrames for representing natural
language data, plus a library of functions for working with these extension
types.

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

To install Text Extensions for Pandas, just run

```
pip install text_extensions_for_pandas
```

from your Python 3.7 or later environment.

Depending on your use case, you may also need the following additional
packages:
* `spacy` (for SpaCy support)
* `transformers` (for 
* `ibm_watson` (for IBM Watson support)


## Documentation

For examples of how to use the library, take a look at the notebooks in 
[this directory](https://github.com/CODAIT/text-extensions-for-pandas/tree/master/notebooks).

API documentation can be found at [https://readthedocs.org/projects/text-extensions-for-pandas/](https://readthedocs.org/projects/text-extensions-for-pandas/).

## Source Code

The source code for Text Extensions for Pandas is available at [https://github.com/CODAIT/text-extensions-for-pandas](https://github.com/CODAIT/text-extensions-for-pandas).

We welcome code and documentation contributions!  See the [README file](https://github.com/CODAIT/text-extensions-for-pandas/blob/master/README.md#contributing) 
for more information on contributing.






