from sympy import MatrixSymbol, Q, ZeroMatrix
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
