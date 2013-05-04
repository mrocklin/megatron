from megatron.util import (unique, merge, groupby, remove)


def test_unique():
    assert tuple(unique((1, 3, 1, 2))) == (1, 3, 2)
    assert tuple(unique((1, 3, 1, 2), key=lambda x: x%2)) == (1, 2)

def test_merge():
    assert merge({1: 2}, {2: 3}, {3: 4, 1: 5}) == {1: 5, 2: 3, 3: 4}

def test_groupby():
    d = groupby(lambda x: x%2, range(10))
    assert set(d.keys()) == set((0, 1))
    assert set(d[0]) == set((0, 2, 4, 6, 8))
    assert set(d[1]) == set((1, 3, 5, 7, 9))

def test_remove():
    assert remove(str.islower, 'AaBb') == ['A', 'B']
