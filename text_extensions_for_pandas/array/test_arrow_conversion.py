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

import unittest

import numpy as np
import numpy.testing as npt
import pyarrow as pa

from text_extensions_for_pandas.array.arrow_conversion import ArrowTensorArray


def _ipc_write_batches(batches):
    assert len(batches) > 0
    stream = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(stream, batches[0].schema)
    for batch in batches:
        writer.write_batch(batch)
    writer.close()
    return stream.getvalue()


def _ipc_read_batches(buf):
    reader = pa.RecordBatchStreamReader(buf)
    return [batch for batch in reader]


def _roundtrip_batch(record_batch):
    buf = _ipc_write_batches([record_batch])
    batches = _ipc_read_batches(buf)
    assert len(batches) == 1
    return batches[0]


def _roundtrip_table(table):
    batches = table.to_batches()
    buf = _ipc_write_batches(batches)
    result_batches = _ipc_read_batches(buf)
    return pa.Table.from_batches(result_batches)


class TestArrowTensor(unittest.TestCase):

    def test_numpy_roundtrip(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        arr = ArrowTensorArray.from_numpy(x)
        self.assertEqual(len(arr), 3)
        batch = pa.RecordBatch.from_arrays([arr], ["batched_tensor"])
        result_batch = _roundtrip_batch(batch)
        result_arr = result_batch.column(0)
        result = result_arr.to_numpy()
        npt.assert_array_equal(x, result)

    def test_list_of_numpy_roundtrip(self):
        x = [np.array([i, i * 2]) for i in range(5)]
        arr = ArrowTensorArray.from_numpy(x)
        batch = pa.RecordBatch.from_arrays([arr], ["batched_tensor"])
        result_batch = _roundtrip_batch(batch)
        result_arr = result_batch.column(0)
        result = result_arr.to_numpy()
        expected = np.stack(x)
        npt.assert_array_equal(expected, result)

    def test_batch_size(self):
        x = [np.array([i, i * 2]) for i in range(6)]
        arr_iter = ArrowTensorArray.from_numpy(x, batch_size=3)
        result_obj_list = []
        for arr in arr_iter:
            batch = pa.RecordBatch.from_arrays([arr], ["batched_tensor"])
            result_batch = _roundtrip_batch(batch)
            result_arr = result_batch.column(0)
            result_obj_list.append(result_arr.to_numpy())
        self.assertEqual(len(result_obj_list), 2)
        result = np.concatenate(result_obj_list)
        expected = np.stack(x)
        npt.assert_array_equal(expected, result)

    def test_ndarray_dict(self):
        obj = {'a': [np.array([i, i * 2]) for i in range(10)],
               'b': [np.array([i, i * i]) for i in range(10)]}
        table = ArrowTensorArray.from_numpy(obj)
        result_table = _roundtrip_table(table)
        results_a = [chunk.to_numpy() for chunk in result_table.column('a').iterchunks()]
        results_b = [chunk.to_numpy() for chunk in result_table.column('b').iterchunks()]
        result_a = np.concatenate(results_a)
        result_b = np.concatenate(results_b)
        self.assertTrue(np.array_equal(obj['a'], result_a))
        self.assertTrue(np.array_equal(obj['b'], result_b))
