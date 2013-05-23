from computations.matrices.examples.linregress import (c, assumptions, X, y,
        beta)
from sympy import assuming
from megatron.core import compile

def test_compile():
    result = compile([X, y], [beta], *assumptions)
    assert set(map(type, result.computations)) == set(map(type, c.computations))

from computations.matrices.fortran.core import build
def test_numeric():
    with assuming(*assumptions):
        result = compile([X, y], [beta])
        f = build(result, [X, y], [beta],
                    modname='linregress', filename='linregress.f90')

    assert callable(f)
    print f.__doc__
    import numpy as np
    nX = np.asarray([[2, 1], [1, 2]], dtype='float64').reshape((2, 2))
    ny = np.ones(2)

    mX = np.matrix(nX)
    my = np.matrix(ny).T
    expected = np.linalg.solve(mX.T*mX, mX.T*my)
    assert np.allclose(expected, f(nX, ny))
