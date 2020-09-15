


Introduction
**************************************************

This page describes the Python API for the `Text Extensions for Pandas`_ library.

.. _`Text Extensions for Pandas`: https://github.com/CODAIT/text-extensions-for-pandas

Python Classes and Functions
**************************************************

Extension Types
===============

Text Extensions for Pandas includes extension types for representing
`spans` and `tensors` inside Pandas DataFrames.
This section describes the Python classes that implement
these types.

.. automodule::   text_extensions_for_pandas

Input and Output
=================

Text Extensions for Pandas includes functionality for converting the outputs of
common NLP libraries into Pandas DataFrames. This section describes these
I/O-related integrations.

In addition to the functionality described in this section, our extension types
also support Pandas' native serialization via `Apache Arrow`_, including the
`to_feather`_ and `read_feather`_ methods for binary file I/O.

.. _`Apache Arrow`: https://arrow.apache.org/docs/python/pandas.html
.. _`to_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_feather.html
.. _`read_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html


``io.watson.nlu`` Module: IBM Watson Natural Language Understanding
-------------------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.watson.nlu

``io.watson.tables`` Module: IBM Watson Discovery Table Understanding
---------------------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.watson.tables

``io.conll`` Module: CoNLL-2003 File Format
--------------------------------------------

.. automodule:: text_extensions_for_pandas.io.conll

``io.spacy`` Module: Pandas APIs for SpaCy Data Structures
-----------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.spacy

``io.bert`` Module: Support for BERT (and similar) embeddings
---------------------------------------------------------------------

.. automodule:: text_extensions_for_pandas.io.bert


``spanner`` Module: Spanner Algebra
=====================================

.. automodule::   text_extensions_for_pandas.spanner


``jupyter`` Module: Support for Jupyter Notebooks
=================================================================

.. automodule:: text_extensions_for_pandas.jupyter




Indices and tables
**************************************************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 5