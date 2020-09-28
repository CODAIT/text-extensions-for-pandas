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

import numpy as np
import pyarrow as pa

from text_extensions_for_pandas.array.span import SpanArray
from text_extensions_for_pandas.array.token_span import TokenSpanArray
from text_extensions_for_pandas.array.tensor import TensorArray


class ArrowSpanType(pa.PyExtensionType):
    """
    PyArrow extension type definition for conversions to/from Span columns
    """

    TARGET_TEXT_KEY = b"target_text"  # metadata key/value gets serialized to bytes
    BEGINS_NAME = "char_begins"
    ENDS_NAME = "char_ends"

    def __init__(self, index_dtype, target_text):
        """
        Create an instance of a Span data type with given index type and
        target text that will be stored in Field metadata.

        :param index_dtype:
        :param target_text:
        """
        assert pa.types.is_integer(index_dtype)

        # Store target text as field metadata
        metadata = {self.TARGET_TEXT_KEY: target_text}

        fields = [
            pa.field(self.BEGINS_NAME, index_dtype, metadata=metadata),
            pa.field(self.ENDS_NAME, index_dtype)
        ]

        pa.PyExtensionType.__init__(self, pa.struct(fields))

    def __reduce__(self):
        index_dtype = self.storage_type[self.BEGINS_NAME].type
        metadata = self.storage_type[self.BEGINS_NAME].metadata
        return ArrowSpanType, (index_dtype, metadata)


class ArrowTokenSpanType(pa.PyExtensionType):
    """
    PyArrow extension type definition for conversions to/from TokenSpan columns
    """

    TARGET_TEXT_KEY = ArrowSpanType.TARGET_TEXT_KEY
    BEGINS_NAME = "token_begins"
    ENDS_NAME = "token_ends"

    def __init__(self, index_dtype, target_text, num_char_span_splits):
        """
        Create an instance of a TokenSpan data type with given index type and
        target text that will be stored in Field metadata.

        :param index_dtype:
        :param target_text:
        """
        assert pa.types.is_integer(index_dtype)
        self.num_char_span_splits = num_char_span_splits

        # Store target text as field metadata
        metadata = {self.TARGET_TEXT_KEY: target_text}

        token_span_fields = [
            pa.field(self.BEGINS_NAME, index_dtype, metadata=metadata),
            pa.field(self.ENDS_NAME, index_dtype),
        ]

        # Span arrays fit into single fields
        if num_char_span_splits == 0:
            char_span_fields = [
                pa.field(ArrowSpanType.BEGINS_NAME, index_dtype),
                pa.field(ArrowSpanType.ENDS_NAME, index_dtype)
            ]
        # Store splits of Span as multiple fields
        else:
            char_span_fields = []
            for i in range(num_char_span_splits):
                n = "_{}".format(i)
                begin_field = pa.field(ArrowSpanType.BEGINS_NAME + n, index_dtype)
                end_field = pa.field(ArrowSpanType.ENDS_NAME + n, index_dtype)
                char_span_fields.extend([begin_field, end_field])

        fields = token_span_fields + char_span_fields

        pa.PyExtensionType.__init__(self, pa.struct(fields))

    def __reduce__(self):
        index_dtype = self.storage_type[self.BEGINS_NAME].type
        metadata = self.storage_type[self.BEGINS_NAME].metadata
        target_text = metadata[self.TARGET_TEXT_KEY].decode()
        num_char_span_splits = self.num_char_span_splits
        return ArrowTokenSpanType, (index_dtype, target_text, num_char_span_splits)


def span_to_arrow(char_span: SpanArray) -> pa.ExtensionArray:
    """
    Convert a SpanArray to a pyarrow.ExtensionArray with a type
    of ArrowSpanType and struct as the storage type. The resulting
    extension array can be serialized and transferred with standard
    Arrow protocols.

    :param char_span: A SpanArray to be converted
    :return: pyarrow.ExtensionArray containing Span data
    """
    # Create array for begins, ends
    begins_array = pa.array(char_span.begin)
    ends_array = pa.array(char_span.end)

    typ = ArrowSpanType(begins_array.type, char_span.target_text)
    fields = list(typ.storage_type)

    storage = pa.StructArray.from_arrays([begins_array, ends_array], fields=fields)

    return pa.ExtensionArray.from_storage(typ, storage)


def arrow_to_span(extension_array: pa.ExtensionArray) -> SpanArray:
    """
    Convert a pyarrow.ExtensionArray with type ArrowSpanType to
    a SpanArray.

    :param extension_array: pyarrow.ExtensionArray with type ArrowSpanType
    :return: SpanArray
    """
    if isinstance(extension_array, pa.ChunkedArray):
        if extension_array.num_chunks > 1:
            raise ValueError("Only pyarrow.Array with a single chunk is supported")
        extension_array = extension_array.chunk(0)

    assert pa.types.is_struct(extension_array.storage.type)

    # Get target text from the begins field metadata and decode string
    metadata = extension_array.storage.type[ArrowSpanType.BEGINS_NAME].metadata
    target_text = metadata[ArrowSpanType.TARGET_TEXT_KEY]
    if isinstance(target_text, bytes):
        target_text = target_text.decode()

    # Get the begins/ends pyarrow arrays
    begins_array = extension_array.storage.field(ArrowSpanType.BEGINS_NAME)
    ends_array = extension_array.storage.field(ArrowSpanType.ENDS_NAME)

    # Zero-copy convert arrays to numpy
    begins = begins_array.to_numpy()
    ends = ends_array.to_numpy()

    return SpanArray(target_text, begins, ends)


def token_span_to_arrow(token_span: TokenSpanArray) -> pa.ExtensionArray:
    """
    Convert a TokenSpanArray to a pyarrow.ExtensionArray with a type
    of ArrowTokenSpanType and struct as the storage type. The resulting
    extension array can be serialized and transferred with standard
    Arrow protocols.

    :param token_span: A TokenSpanArray to be converted
    :return: pyarrow.ExtensionArray containing TokenSpan data
    """
    # Create arrays for begins/ends
    token_begins_array = pa.array(token_span.begin_token)
    token_ends_array = pa.array(token_span.end_token)
    token_span_arrays = [token_begins_array, token_ends_array]

    num_char_span_splits = 0

    # If TokenSpan arrays have greater length than Span arrays, pad Span
    if len(token_span.begin_token) > len(token_span.tokens.begin):

        padding = np.zeros(len(token_span.begin_token) - len(token_span.tokens.begin),
                           token_span.tokens.begin.dtype)

        isnull = np.append(np.full(len(token_span.tokens.begin), False), np.full(len(padding), True))
        char_begins_padded = np.append(token_span.tokens.begin, padding)
        char_ends_padded = np.append(token_span.tokens.end, padding)
        char_begins_array = pa.array(char_begins_padded, mask=isnull)
        char_ends_array = pa.array(char_ends_padded, mask=isnull)
        char_span_arrays = [char_begins_array, char_ends_array]

    # If TokenSpan arrays have less length than Span arrays, split Span into multiple arrays
    elif len(token_span.begin_token) < len(token_span.tokens.begin):

        char_begins_array = pa.array(token_span.tokens.begin)
        char_ends_array = pa.array(token_span.tokens.end)

        char_span_arrays = []
        while len(char_begins_array) >= len(token_begins_array):
            char_begins_split = char_begins_array[:len(token_begins_array)]
            char_ends_split = char_ends_array[:len(token_ends_array)]

            char_span_arrays.extend([char_begins_split, char_ends_split])
            num_char_span_splits += 1

            char_begins_array = char_begins_array[len(token_begins_array):]
            char_ends_array = char_ends_array[len(token_ends_array):]

        # Pad the final split
        if len(char_begins_array) > 0:
            padding = np.zeros(len(token_begins_array) - len(char_begins_array),
                               token_span.tokens.begin.dtype)
            isnull = np.append(np.full(len(char_begins_array), False), np.full(len(padding), True))
            char_begins_padded = np.append(char_begins_array.to_numpy(), padding)
            char_ends_padded = np.append(char_ends_array.to_numpy(), padding)
            char_begins_split = pa.array(char_begins_padded, mask=isnull)
            char_ends_split = pa.array(char_ends_padded, mask=isnull)
            char_span_arrays.extend([char_begins_split, char_ends_split])
            num_char_span_splits += 1

    # TokenSpan arrays are equal length to Span arrays
    else:
        char_begins_array = pa.array(token_span.tokens.begin)
        char_ends_array = pa.array(token_span.tokens.end)
        char_span_arrays = [char_begins_array, char_ends_array]

    typ = ArrowTokenSpanType(token_begins_array.type, token_span.target_text, num_char_span_splits)
    fields = list(typ.storage_type)

    storage = pa.StructArray.from_arrays(token_span_arrays + char_span_arrays, fields=fields)

    return pa.ExtensionArray.from_storage(typ, storage)


def arrow_to_token_span(extension_array: pa.ExtensionArray) -> TokenSpanArray:
    """
    Convert a pyarrow.ExtensionArray with type ArrowTokenSpanType to
    a TokenSpanArray.

    :param extension_array: pyarrow.ExtensionArray with type ArrowTokenSpanType
    :return: TokenSpanArray
    """
    if isinstance(extension_array, pa.ChunkedArray):
        if extension_array.num_chunks > 1:
            raise ValueError("Only pyarrow.Array with a single chunk is supported")
        extension_array = extension_array.chunk(0)

    assert pa.types.is_struct(extension_array.storage.type)

    # Get target text from the begins field metadata and decode string
    metadata = extension_array.storage.type[ArrowTokenSpanType.BEGINS_NAME].metadata
    target_text = metadata[ArrowSpanType.TARGET_TEXT_KEY]
    if isinstance(target_text, bytes):
        target_text = target_text.decode()

    # Get the begins/ends pyarrow arrays
    token_begins_array = extension_array.storage.field(ArrowTokenSpanType.BEGINS_NAME)
    token_ends_array = extension_array.storage.field(ArrowTokenSpanType.ENDS_NAME)

    # Check if CharSpans have been split
    num_char_span_splits = extension_array.type.num_char_span_splits
    if num_char_span_splits > 0:
        char_begins_splits = []
        char_ends_splits = []
        for i in range(num_char_span_splits):
            char_begins_splits.append(
                extension_array.storage.field(ArrowSpanType.BEGINS_NAME + "_{}".format(i)))
            char_ends_splits.append(
                extension_array.storage.field(ArrowSpanType.ENDS_NAME + "_{}".format(i)))
        char_begins_array = pa.concat_arrays(char_begins_splits)
        char_ends_array = pa.concat_arrays(char_ends_splits)
    else:
        char_begins_array = extension_array.storage.field(ArrowSpanType.BEGINS_NAME)
        char_ends_array = extension_array.storage.field(ArrowSpanType.ENDS_NAME)

    # Remove any trailing padding
    if char_begins_array.null_count > 0:
        char_begins_array = char_begins_array[:-char_begins_array.null_count]
        char_ends_array = char_ends_array[:-char_ends_array.null_count]

    # Zero-copy convert arrays to numpy
    token_begins = token_begins_array.to_numpy()
    token_ends = token_ends_array.to_numpy()
    char_begins = char_begins_array.to_numpy()
    char_ends = char_ends_array.to_numpy()

    # Create the SpanArray, then the TokenSpanArray
    char_span = SpanArray(target_text, char_begins, char_ends)
    return TokenSpanArray(char_span, token_begins, token_ends)


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
