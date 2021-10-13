Pandas Extension Types
***********************

Text Extensions for Pandas includes extension types for representing
*spans* and *tensors* inside Pandas DataFrames.
This section describes the Python classes that implement
these types.



Span Extension Type
-------------------------------------------------------------------

The :class:`SpanDtype` extension data type efficiently stores span data
in a Pandas Series.
Each span is represented by begin and end character offsets
into a target document.
We use dense NumPy arrays for efficient internal storage.

.. autoclass:: text_extensions_for_pandas.SpanDtype

SpanArray Class: Store spans in a Pandas Series
=================================================
.. autoclass:: text_extensions_for_pandas.SpanArray
   :members:

   .. autoclasstoc::

Span Class: Object to represent a single span
==================================================
.. autoclass:: text_extensions_for_pandas.Span
   :members:

   .. autoclasstoc::


Token-Based Span Extension Type
-------------------------------------------------------------------

The :class:`TokenSpanDtype` extension data type is similar to 
:class:`SpanDtype`, except that it represents spans using 
begin and end offsets into the **tokens** of a target document.
These tokens are stored in a (shared) :class:`SpanArray` object.

.. autoclass:: text_extensions_for_pandas.TokenSpanDtype

TokenSpanArray Class: Store token-based spans in a Pandas Series
=================================================================
.. autoclass:: text_extensions_for_pandas.TokenSpanArray
   :members:

   .. autoclasstoc::

TokenSpan Class: Object to represent a single token-based span
=================================================================
.. autoclass:: text_extensions_for_pandas.TokenSpan
   :members:

   .. autoclasstoc::


Tensor Extension Type
-------------------------------------------------------------------

The :class:`TensorDtype` extension data type is efficiently stores
tensors in the rows of a Pandas Series.
For efficiency, we store all of the tensors in a Series in a single
NumPy array.

.. autoclass:: text_extensions_for_pandas.TensorDtype

TensorArray Class: Store tensors in a Pandas Series
=============================================================
.. autoclass:: text_extensions_for_pandas.TensorArray
   :members:

   .. autoclasstoc::

TensorElement Class: Object to represent a single tensor
=============================================================
.. autoclass:: text_extensions_for_pandas.TensorElement
   :members:

   .. autoclasstoc::


