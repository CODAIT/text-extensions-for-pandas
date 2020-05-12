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
# arrow_compat.py
#
# Part of text_extensions_for_pandas
#
# Provide Arrow compatible classes for serializing to pyarrow.
#

import numpy as np
import pyarrow as pa

from text_extensions_for_pandas.array import CharSpanArray


class ArrowCharSpanType(pa.PyExtensionType):
    """
    PyArrow extension type definition for conversions to/from CharSpan columns
    """

    TARGET_TEXT_KEY = b"target_text"  # metadata key/value gets serialized to bytes
    BEGINS_NAME = "begins"
    ENDS_NAME = "ends"

    def __init__(self, index_dtype, target_text):
        """
        Create an instance of a CharSpan data type with given index type and
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
        return ArrowCharSpanType, ()


def char_span_to_arrow(char_span):
    """
    Convert a CharSpanArray to a pyarrow.ExtensionArray with a type
    of ArrowCharSpanType and struct as the storage type. The resulting
    extension array can be serialized and transferred with standard
    Arrow protocols.

    :param char_span: A CharSpanArray to be converted
    :return: pyarrow.ExtensionArray containing CharSpan data
    """
    # Create array for begins, ends
    begins_array = pa.array(char_span.begin)
    ends_array = pa.array(char_span.end)

    typ = ArrowCharSpanType(begins_array.type, char_span.target_text)
    data_fields = list(typ.storage_type)

    storage = pa.StructArray.from_arrays([begins_array, ends_array], fields=data_fields)

    return pa.ExtensionArray.from_storage(typ, storage)


def arrow_to_char_span(extension_array):
    """
    Convert a pyarrow.ExtensionArray with type ArrowCharSpanType to
    a CharSpanArray.

    :param extension_array: pyarrow.ExtensionArray with type ArrowCharSpanType
    :return: CharSpanArray
    """
    if isinstance(extension_array, pa.ChunkedArray):
        if extension_array.num_chunks > 1:
            raise ValueError("Only pyarrow.Array with a single chunk is supported")
        extension_array = extension_array.chunk(0)

    assert pa.types.is_struct(extension_array.storage.type)

    # Get target text from the begins field metadata and decode string
    metadata = extension_array.storage.type[ArrowCharSpanType.BEGINS_NAME].metadata
    target_text = metadata[ArrowCharSpanType.TARGET_TEXT_KEY]
    if isinstance(target_text, bytes):
        target_text = target_text.decode()

    # Get the begins/ends pyarrow arrays
    begins_array = extension_array.storage.field(ArrowCharSpanType.BEGINS_NAME)
    ends_array = extension_array.storage.field(ArrowCharSpanType.ENDS_NAME)

    # Zero-copy convert arrays to numpy
    begins = begins_array.to_numpy()
    ends = ends_array.to_numpy()

    return CharSpanArray(target_text, begins, ends)


class ArrowTensorType(pa.PyExtensionType):
    """
    pyarrow ExtensionType definition for TensorType
    """
    def __init__(self, shape, pyarrow_dtype):
        self._shape = shape
        pa.PyExtensionType.__init__(self, pa.list_(pyarrow_dtype))

    def __reduce__(self):
        return ArrowTensorType, (self._shape, self.storage_type.value_type)

    @property
    def shape(self):
        return self._shape


class ArrowTensorArray(object):
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
            total_num_elements = obj.size
            num_elements = 1 if len(obj.shape) == 1 else np.prod(obj.shape[1:])

            child_buf = pa.py_buffer(obj)
            child_array = pa.Array.from_buffers(pa_dtype, total_num_elements, [None, child_buf])

            offset_buf = pa.py_buffer(np.int32([i * num_elements for i in range(batch_size + 1)]))

            storage = pa.Array.from_buffers(pa.list_(pa_dtype), batch_size,
                                            [None, offset_buf], children=[child_array])

            typ = ArrowTensorType(obj.shape, pa_dtype)
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

    @staticmethod
    def to_numpy(pa_ext_array):
        if isinstance(pa_ext_array, pa.ChunkedArray):
            if pa_ext_array.num_chunks > 1:
                raise ValueError("Only pyarrow.Column with single array chunk is supported")
            pa_ext_array = pa_ext_array.chunk(0)
        ext_type = pa_ext_array.type
        ext_list_type = pa_ext_array.storage.type
        # assert ext_list_type is pa.list_
        ext_dtype = ext_list_type.value_type.to_pandas_dtype()
        buf = pa_ext_array.storage.buffers()[3]
        return np.ndarray(ext_type.shape, buffer=buf, dtype=ext_dtype)
