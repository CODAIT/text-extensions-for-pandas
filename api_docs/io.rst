Input and Output
*******************************************************************

Text Extensions for Pandas includes functionality for converting the outputs of
common NLP libraries into Pandas DataFrames. This section describes these
I/O-related integrations.

In addition to the functionality described in this section, our extension types
also support Pandas' native serialization via `Apache Arrow`_, including the
`to_feather`_ and `read_feather`_ methods for binary file I/O.

.. _`Apache Arrow`: https://arrow.apache.org/docs/python/pandas.html
.. _`to_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_feather.html
.. _`read_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html


IBM Watson Natural Language Understanding
-------------------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.watson.nlu
   :members:
   :undoc-members:

IBM Watson Discovery Table Understanding
---------------------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.watson.tables
   :members:
   :undoc-members:

CoNLL-2003 and CoNLL-U File Formats
--------------------------------------------

.. automodule:: text_extensions_for_pandas.io.conll
   :members:
   :undoc-members:

Pandas APIs for SpaCy Data Structures
-----------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.spacy
   :members:
   :undoc-members:

Support for BERT and similar language models
---------------------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.bert
   :members:
   :undoc-members:

