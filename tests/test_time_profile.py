import arkouda as ak
import pytest
from ampere import AttributionEngine, Rank, Node, Run, Ensemble

# A simple sequential rank:
# main:    [0.0, 10.0]  (Depth 0)
#   child: [2.0, 5.0]   (Depth 1)

def test_time_profile_inclusive():
    ak.connect(server="localhost", port=5555)
    
    # We must match Rank's expected dataframe schema
    calls = {
        'Start Time': ak.array([0.0, 2.0]),
        'End Time':   ak.array([10.0, 5.0]),
        'Name':       ak.array(['main', 'child']),
        'Depth':      ak.array([0, 1]),
    }
    
    r = Rank("test_node", "rank_0", calls)
    
    # Calculate Inclusive time
    res = AttributionEngine.compute(None, [r], strategy='inclusive')
    df = res["rank_0"]
    
    vals = df['Value'].to_ndarray()
    assert vals[0] == 10.0  # Main spans [0, 10]
    assert vals[1] == 3.0   # Child spans [2, 5]
    
def test_time_profile_exclusive():
    # Arkouda assumes already connected
    calls = {
        'Start Time': ak.array([0.0, 2.0]),
        'End Time':   ak.array([10.0, 5.0]),
        'Name':       ak.array(['main', 'child']),
        'Depth':      ak.array([0, 1]),
    }
    
    r = Rank("test_node", "rank_0", calls)
    
    # Calculate Exclusive time
    res = AttributionEngine.compute(None, [r], strategy='exclusive')
    df = res["rank_0"]
    
    vals = df['Value'].to_ndarray()
    
    # Main gets [0, 2] and [5, 10] = 2.0 + 5.0 = 7.0
    assert vals[0] == 7.0
    
    # Child gets all of [2, 5] = 3.0
    assert vals[1] == 3.0
    
if __name__ == '__main__':
    pytest.main([__file__])
