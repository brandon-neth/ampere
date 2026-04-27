import sys
import os
import numpy as np

# Ensure local source is first in path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

# Patch numpy to prevent arkouda import errors if it sneaks through
if not hasattr(np, 'bool'):
    np.bool = bool

from unittest.mock import MagicMock

class MockPdArray(np.ndarray):
    def to_ndarray(self):
        return self

def as_mock_pdarray(arr):
    return np.asarray(arr).view(MockPdArray)

# Mock Arkouda (ak) module thoroughly
ak_mock = MagicMock()
ak_mock.float64 = np.float64
ak_mock.int64 = np.int64
ak_mock.zeros = lambda size, dtype=None: as_mock_pdarray(np.zeros(size, dtype=dtype))
ak_mock.ones = lambda size, dtype=None: as_mock_pdarray(np.ones(size, dtype=dtype))
ak_mock.where = lambda c, x, y: as_mock_pdarray(np.where(c, x, y))
ak_mock.cumsum = lambda arr: as_mock_pdarray(np.cumsum(arr))
ak_mock.concatenate = lambda arrs: as_mock_pdarray(np.concatenate(arrs))
ak_mock.searchsorted = lambda arr, v, side='left': as_mock_pdarray(np.searchsorted(arr, v, side=side))
ak_mock.unique = lambda arr: as_mock_pdarray(np.unique(arr))
ak_mock.sort = lambda arr: as_mock_pdarray(np.sort(arr))
ak_mock.array = lambda arr: as_mock_pdarray(np.array(arr))

class GroupByMock:
    def __init__(self, arr):
        self.arr = arr
    def aggregate(self, vals, op):
        if op == 'sum':
            unique_keys = np.unique(self.arr)
            sums = np.zeros_like(unique_keys)
            for i, k in enumerate(unique_keys):
                sums[i] = np.sum(vals[self.arr == k])
            return as_mock_pdarray(unique_keys), as_mock_pdarray(sums)
        return as_mock_pdarray(np.unique(self.arr)), as_mock_pdarray(np.unique(self.arr))

ak_mock.GroupBy = GroupByMock

def ak_DataFrame_mock(d):
    class DF:
        def __init__(self, data):
            self.data = data
            self.keys = lambda: list(data.keys())
            self.size = len(list(data.values())[0]) if data else 0
        def __getitem__(self, key):
            return self.data[key]
    return DF(d)

ak_mock.DataFrame = ak_DataFrame_mock

# Inject the mock BEFORE importing ampere
sys.modules['arkouda'] = ak_mock

from ampere import AttributionEngine, Rank

def test_time_profile_inclusive():
    calls = {
        'Start Time': as_mock_pdarray([0.0, 2.0]),
        'End Time':   as_mock_pdarray([10.0, 5.0]),
        'Name':       as_mock_pdarray(['main', 'child']),
        'Depth':      as_mock_pdarray([0, 1]),
    }
    r = Rank("test_node", "rank_0", calls)
    res = AttributionEngine.compute(None, [r], strategy='inclusive')
    df = res["rank_0"]
    vals = df['Value']
    assert vals[0] == 10.0
    assert vals[1] == 3.0
    print("test_time_profile_inclusive PASSED")

def test_time_profile_exclusive():
    calls = {
        'Start Time': as_mock_pdarray([0.0, 2.0]),
        'End Time':   as_mock_pdarray([10.0, 5.0]),
        'Name':       as_mock_pdarray(['main', 'child']),
        'Depth':      as_mock_pdarray([0, 1]),
    }
    r = Rank("test_node", "rank_0", calls)
    res = AttributionEngine.compute(None, [r], strategy='exclusive')
    df = res["rank_0"]
    vals = df['Value']
    assert vals[0] == 7.0
    assert vals[1] == 3.0
    print("test_time_profile_exclusive PASSED")

if __name__ == '__main__':
    test_time_profile_inclusive()
    test_time_profile_exclusive()
