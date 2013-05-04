from sympy import MatrixSymbol, Q, ZeroMatrix, assuming
n, m = 3, 2
X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)
beta = (X.T*X).I * X.T*y

from computations.matrices.blas import SYRK, GEMM
from computations.matrices.lapack import POSV
c = (POSV(X.T*X, X.T*y)
   + SYRK(1.0, X.T, 0.0, ZeroMatrix(m, m))
   + GEMM(1.0, X.T, y, 0.0, ZeroMatrix(n, 1)))

from megatron.core import compile
def test_compile():
    result = compile([X, y], [beta], Q.fullrank(X))
    assert set(map(type, result.computations)) == set(map(type, c.computations))

from computations.matrices.fortran.core import build
def test_numeric():
    with assuming(Q.fullrank(X), Q.real_elements(X), Q.real_elements(y)):
        result = compile([X, y], [beta])
        f = build(result, [X, y], [beta],
                    modname='linregress', filename='linregress.f90')

    assert callable(f)
    import numpy as np
    nX = np.asarray([[2, 1], [1, 2]], dtype='float64').reshape((2, 2))
    ny = np.ones(2)

    mX = np.matrix(nX)
    my = np.matrix(ny).T
    expected = np.linalg.solve(mX.T*mX, mX.T*my)
    assert np.allclose(expected, f(nX, ny))
