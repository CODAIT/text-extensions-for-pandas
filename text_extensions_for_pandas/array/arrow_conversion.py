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

#
# arrow_conversion.py
#
# Part of text_extensions_for_pandas
#
# Provide Arrow compatible classes for serializing to pyarrow.
#
from distutils.version import LooseVersion

import numpy as np
import pyarrow as pa

from text_extensions_for_pandas.array.span import SpanArray
from text_extensions_for_pandas.array.token_span import TokenSpanArray, _EMPTY_SPAN_ARRAY_SINGLETON
from text_extensions_for_pandas.array.tensor import TensorArray
from text_extensions_for_pandas.array.string_table import StringTable


class ArrowSpanType(pa.PyExtensionType):
    """
    PyArrow extension type definition for conversions to/from Span columns
    """
    BEGINS_NAME = "span_begins"
    ENDS_NAME = "span_ends"
    TARGET_TEXT_DICT_NAME = "target_text"

    def __init__(self, index_dtype, target_text_dict_dtype):
        """
        Create an instance of a Span data type with given index type and
        target text dictionary type. The dictionary type will hold target text ids
        that map to a dictionary of document target texts.

        :param index_dtype: type for the begin, end index arrays
        :param target_text_dict_dtype: type for the target text dictionary array
        """
        assert pa.types.is_integer(index_dtype)
        assert pa.types.is_dictionary(target_text_dict_dtype)

        fields = [
            pa.field(self.BEGINS_NAME, index_dtype),
            pa.field(self.ENDS_NAME, index_dtype),
            pa.field(self.TARGET_TEXT_DICT_NAME, target_text_dict_dtype)
        ]

        pa.PyExtensionType.__init__(self, pa.struct(fields))

    def __reduce__(self):
        index_dtype = self.storage_type[self.BEGINS_NAME].type
        target_text_dict_dtype = self.storage_type[self.TARGET_TEXT_DICT_NAME].type
        return ArrowSpanType, (index_dtype, target_text_dict_dtype)


class ArrowTokenSpanType(pa.PyExtensionType):
    """
    PyArrow extension type definition for conversions to/from TokenSpan columns
    """

    BEGINS_NAME = "token_begins"
    ENDS_NAME = "token_ends"
    TOKENS_NAME = "tokens"

    def __init__(self, index_dtype, token_dict_dtype):
        """
        Create an instance of a TokenSpan data type with given index type and
        target text that will be stored in Field metadata.

        :param index_dtype: type for the begin, end index arrays
        :param token_dict_dtype: type for the tokens dictionary array
        """
        assert pa.types.is_integer(index_dtype)
        assert pa.types.is_dictionary(token_dict_dtype)

        fields = [
            pa.field(self.BEGINS_NAME, index_dtype),
            pa.field(self.ENDS_NAME, index_dtype),
            pa.field(self.TOKENS_NAME, token_dict_dtype),
        ]

        pa.PyExtensionType.__init__(self, pa.struct(fields))

    def __reduce__(self):
        index_dtype = self.storage_type[self.BEGINS_NAME].type
        token_dict_dtype = self.storage_type[self.TOKENS_NAME].type
        return ArrowTokenSpanType, (index_dtype, token_dict_dtype)


def span_to_arrow(char_span: SpanArray) -> pa.ExtensionArray:
    """
    Convert a SpanArray to a pyarrow.ExtensionArray with a type
    of ArrowSpanType and struct as the storage type. The resulting
    extension array can be serialized and transferred with standard
    Arrow protocols.

    :param char_span: A SpanArray to be converted
    :return: pyarrow.ExtensionArray containing Span data
    """
    if LooseVersion(pa.__version__) < LooseVersion("2.0.0"):
        raise NotImplementedError("Arrow serialization for SpanArray is not supported with "
                                  "PyArrow versions < 2.0.0")
    # Create array for begins, ends
    begins_array = pa.array(char_span.begin)
    ends_array = pa.array(char_span.end)

    # Create a dictionary array from StringTable used in this span
    dictionary = pa.array([char_span._string_table.unbox(s)
                           for s in char_span._string_table.things])
    target_text_dict_array = pa.DictionaryArray.from_arrays(char_span._text_ids, dictionary)

    typ = ArrowSpanType(begins_array.type, target_text_dict_array.type)
    fields = list(typ.storage_type)

    storage = pa.StructArray.from_arrays([begins_array, ends_array, target_text_dict_array], fields=fields)

    return pa.ExtensionArray.from_storage(typ, storage)


def arrow_to_span(extension_array: pa.ExtensionArray) -> SpanArray:
    """
    Convert a pyarrow.ExtensionArray with type ArrowSpanType to
    a SpanArray.

    ..NOTE: Only supported with PyArrow >= 2.0.0

    :param extension_array: pyarrow.ExtensionArray with type ArrowSpanType
    :return: SpanArray
    """
    if LooseVersion(pa.__version__) < LooseVersion("2.0.0"):
        raise NotImplementedError("Arrow serialization for SpanArray is not supported with "
                                  "PyArrow versions < 2.0.0")
    if isinstance(extension_array, pa.ChunkedArray):
        if extension_array.num_chunks > 1:
            raise ValueError("Only pyarrow.Array with a single chunk is supported")
        extension_array = extension_array.chunk(0)

    # NOTE: workaround for bug in parquet reading
    if pa.types.is_struct(extension_array.type):
        index_dtype = extension_array.field(ArrowSpanType.BEGINS_NAME).type
        target_text_dict_dtype = extension_array.field(ArrowSpanType.TARGET_TEXT_DICT_NAME).type
        extension_array = pa.ExtensionArray.from_storage(
            ArrowSpanType(index_dtype, target_text_dict_dtype),
            extension_array)

    assert pa.types.is_struct(extension_array.storage.type)

    # Create target text StringTable and text_ids from dictionary array
    target_text_dict_array = extension_array.storage.field(ArrowSpanType.TARGET_TEXT_DICT_NAME)
    table_texts = target_text_dict_array.dictionary.to_pylist()
    string_table = StringTable.from_things(table_texts)
    text_ids = target_text_dict_array.indices.to_numpy()

    # Get the begins/ends pyarrow arrays
    begins_array = extension_array.storage.field(ArrowSpanType.BEGINS_NAME)
    ends_array = extension_array.storage.field(ArrowSpanType.ENDS_NAME)

    # Zero-copy convert arrays to numpy
    begins = begins_array.to_numpy()
    ends = ends_array.to_numpy()

    return SpanArray((string_table, text_ids), begins, ends)


def token_span_to_arrow(token_span: TokenSpanArray) -> pa.ExtensionArray:
    """
    Convert a TokenSpanArray to a pyarrow.ExtensionArray with a type
    of ArrowTokenSpanType and struct as the storage type. The resulting
    extension array can be serialized and transferred with standard
    Arrow protocols.

    :param token_span: A TokenSpanArray to be converted
    :return: pyarrow.ExtensionArray containing TokenSpan data
    """
    if LooseVersion(pa.__version__) < LooseVersion("2.0.0"):
        raise NotImplementedError("Arrow serialization for TokenSpanArray is not supported with "
                                  "PyArrow versions < 2.0.0")
    # Create arrays for begins/ends
    token_begins_array = pa.array(token_span.begin_token)
    token_ends_array = pa.array(token_span.end_token)

    # Filter out any empty SpanArrays
    non_null_tokens = token_span.tokens[~token_span.isna()]
    assert len(non_null_tokens) > 0

    # Get either single document as a list or use a list of all if multiple docs
    if all([token is non_null_tokens[0] for token in non_null_tokens]):
        tokens_arrays = [non_null_tokens[0]]
        tokens_indices = pa.array([0] * len(token_span.tokens), mask=token_span.isna())
    else:
        raise NotImplementedError("TokenSpan Multi-doc serialization not yet implemented due to "
                                  "ArrowNotImplementedError: Concat with dictionary unification NYI")
        tokens_arrays = non_null_tokens
        tokens_indices = np.zeros_like(token_span.tokens)
        tokens_indices[~token_span.isna()] = range(len(tokens_arrays))
        tokens_indices = pa.array(tokens_indices, mask=token_span.isna())

    # Convert each token SpanArray to Arrow and get as raw storage
    arrow_tokens_arrays = [span_to_arrow(sa).storage for sa in tokens_arrays]

    # Create a list array with each element is an ArrowSpanArray
    # TODO: pyarrow.lib.ArrowNotImplementedError: ('Sequence converter for type dictionary<values=string, indices=int8, ordered=0> not implemented', 'Conversion failed for column ts1 with type TokenSpanDtype')
    #arrow_tokens_arrays_array = pa.array(arrow_tokens_arrays, type=pa.list_(arrow_tokens_arrays[0].type))
    offsets = [0] + [len(a) for a in arrow_tokens_arrays]
    values = pa.concat_arrays(arrow_tokens_arrays)  # TODO: can't concat extension arrays?
    arrow_tokens_arrays_array = pa.ListArray.from_arrays(offsets, values)

    # Create a dictionary array mapping each token SpanArray index used to the list of ArrowSpanArrays
    tokens_dict_array = pa.DictionaryArray.from_arrays(tokens_indices, arrow_tokens_arrays_array)

    typ = ArrowTokenSpanType(token_begins_array.type, tokens_dict_array.type)
    fields = list(typ.storage_type)

    storage = pa.StructArray.from_arrays([token_begins_array, token_ends_array, tokens_dict_array], fields=fields)

    return pa.ExtensionArray.from_storage(typ, storage)


def arrow_to_token_span(extension_array: pa.ExtensionArray) -> TokenSpanArray:
    """
    Convert a pyarrow.ExtensionArray with type ArrowTokenSpanType to
    a TokenSpanArray.

    :param extension_array: pyarrow.ExtensionArray with type ArrowTokenSpanType
    :return: TokenSpanArray
    """
    if LooseVersion(pa.__version__) < LooseVersion("2.0.0"):
        raise NotImplementedError("Arrow serialization for TokenSpanArray is not supported with "
                                  "PyArrow versions < 2.0.0")
    if isinstance(extension_array, pa.ChunkedArray):
        if extension_array.num_chunks > 1:
            raise ValueError("Only pyarrow.Array with a single chunk is supported")
        extension_array = extension_array.chunk(0)

    assert pa.types.is_struct(extension_array.storage.type)

    # Get the begins/ends pyarrow arrays
    token_begins_array = extension_array.storage.field(ArrowTokenSpanType.BEGINS_NAME)
    token_ends_array = extension_array.storage.field(ArrowTokenSpanType.ENDS_NAME)

    # Get the tokens as a dictionary array where indices map to a list of ArrowSpanArrays
    tokens_dict_array = extension_array.storage.field(ArrowTokenSpanType.TOKENS_NAME)
    tokens_indices = tokens_dict_array.indices
    arrow_tokens_arrays_array = tokens_dict_array.dictionary

    # Breakup the list of ArrowSpanArrays and convert back to individual SpanArrays
    tokens_arrays = []
    span_type = None
    for i in range(1, len(arrow_tokens_arrays_array.offsets)):
        start = arrow_tokens_arrays_array.offsets[i - 1].as_py()
        stop = arrow_tokens_arrays_array.offsets[i].as_py()
        arrow_tokens_array = arrow_tokens_arrays_array.values[start:stop]

        # Make an instance of ArrowSpanType
        if span_type is None:
            begins_array = arrow_tokens_array.field(ArrowSpanType.BEGINS_NAME)
            target_text_dict_array = arrow_tokens_array.field(ArrowSpanType.TARGET_TEXT_DICT_NAME)
            span_type = ArrowSpanType(begins_array.type, target_text_dict_array.type)

        # Re-make the Arrow extension type to convert back to a SpanArray
        tokens_array = arrow_to_span(pa.ExtensionArray.from_storage(span_type, arrow_tokens_array))
        tokens_arrays.append(tokens_array)

    # Map the token indices to the actual token SpanArray for each element in the TokenSpanArray
    tokens = [_EMPTY_SPAN_ARRAY_SINGLETON if i is None else tokens_arrays[i]
              for i in tokens_indices.to_pylist()]

    # Zero-copy convert arrays to numpy
    token_begins = token_begins_array.to_numpy()
    token_ends = token_ends_array.to_numpy()

    return TokenSpanArray(tokens, token_begins, token_ends)


class ArrowTensorType(pa.PyExtensionType):
    """
    pyarrow ExtensionType definition for TensorDtype

    :param element_shape: Fixed shape for each tensor element of the array, the
                          outer dimension is the number of elements, or length,
                          of the array.
    """
    def __init__(self, element_shape, pyarrow_dtype):
        self._element_shape = element_shape
        pa.PyExtensionType.__init__(self, pa.list_(pyarrow_dtype))

    def __reduce__(self):
        return ArrowTensorType, (self._element_shape, self.storage_type.value_type)

    @property
    def shape(self):
        return self._element_shape

    def __arrow_ext_class__(self):
        return ArrowTensorArray


class ArrowTensorArray(pa.ExtensionArray):
    """
    A batch of tensors with fixed shape.
    """
    def __init__(self):
        raise TypeError("Do not call ArrowTensorBatch constructor directly, "
                        "use one of the `ArrowTensorBatch.from_*` functions "
                        "instead.")

    @staticmethod
    def from_numpy(obj, batch_size=None):
        """
        Convert a list of numpy.ndarrays with equal shapes or as single
        numpy.ndarray with outer-dim as batch size to a pyarrow.Array
        """
        if isinstance(obj, (list, tuple)):
            if batch_size is not None:
                def list_gen():
                    for i in range(0, len(obj), batch_size):
                        slc = obj[i:i + batch_size]
                        yield ArrowTensorArray.from_numpy(slc, batch_size=None)
                return list_gen()
            elif np.isscalar(obj[0]):
                return pa.array(obj)
            elif isinstance(obj[0], np.ndarray):
                # continue with batched ndarray
                obj = np.stack(obj, axis=0)

        if isinstance(obj, dict):
            names = list(obj.keys())
            arrs = [ArrowTensorArray.from_numpy(obj[k], batch_size=batch_size)
                    for k in names]
            batch = pa.RecordBatch.from_arrays(arrs, names)
            return pa.Table.from_batches([batch])

        elif isinstance(obj, np.ndarray):
            # currently require contiguous ndarray
            if not obj.flags.c_contiguous:
                obj = np.ascontiguousarray(obj)
            pa_dtype = pa.from_numpy_dtype(obj.dtype)
            batch_size = obj.shape[0]
            element_shape = obj.shape[1:]
            total_num_elements = obj.size
            num_elements = 1 if len(obj.shape) == 1 else np.prod(element_shape)

            child_buf = pa.py_buffer(obj)
            child_array = pa.Array.from_buffers(pa_dtype, total_num_elements, [None, child_buf])

            offset_buf = pa.py_buffer(np.int32([i * num_elements for i in range(batch_size + 1)]))

            storage = pa.Array.from_buffers(pa.list_(pa_dtype), batch_size,
                                            [None, offset_buf], children=[child_array])

            typ = ArrowTensorType(element_shape, pa_dtype)
            return pa.ExtensionArray.from_storage(typ, storage)

        elif np.isscalar(obj):
            return pa.array([obj])

        else:
            def iter_gen():
                if batch_size is None:
                    for d in obj:
                        yield ArrowTensorArray.from_numpy(d, batch_size=batch_size)
                else:
                    batch = []
                    for o in obj:
                        batch.append(o)
                        if len(batch) == batch_size:
                            # merge dict
                            if isinstance(batch[0], dict):
                                d = {k: [v] for k, v in batch[0].items()}
                                for i in range(1, len(batch)):
                                    for k, v in batch[i].items():
                                        d[k].append(v)
                                for k in d.keys():
                                    d[k] = np.stack(d[k], axis=0)
                                batch = d
                            yield ArrowTensorArray.from_numpy(batch, batch_size=None)
                            batch = []
            return iter_gen()

    def to_numpy(self):
        shape = (len(self),) + self.type.shape
        buf = self.storage.buffers()[3]
        storage_list_type = self.storage.type
        ext_dtype = storage_list_type.value_type.to_pandas_dtype()
        return np.ndarray(shape, buffer=buf, dtype=ext_dtype)


def arrow_to_tensor_array(extension_array: pa.ExtensionArray) -> TensorArray:
    """
    Convert a pyarrow.ExtensionArray with type ArrowTensorType to a
    TensorArray.

    :param extension_array: pyarrow.ExtensionArray with type ArrowTensorType
    :return: TensorArray
    """

    if isinstance(extension_array, pa.ChunkedArray):
        if extension_array.num_chunks > 1:
            # TODO: look into removing concat and constructing from list w/ shape
            values = np.concatenate([chunk.to_numpy()
                                     for chunk in extension_array.iterchunks()])
        else:
            values = extension_array.chunk(0).to_numpy()
    else:
        values = extension_array.to_numpy()

    return TensorArray(values)
