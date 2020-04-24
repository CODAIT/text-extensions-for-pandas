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
