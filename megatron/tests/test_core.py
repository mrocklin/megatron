from megatron.core import computations_for

from sympy import Symbol, MatrixSymbol, ask, Q, assuming


n = Symbol('n')
X = MatrixSymbol('X', n, n)

def test_simple():
    from computations.matrices.blas import SYRK, GEMM, SYMM
    expr = X * X.T
    assert set(map(type, computations_for(expr))) == set((SYRK, GEMM))

    with assuming(Q.symmetric(X)):
        assert set(map(type, computations_for(expr))) == set((SYRK, GEMM, SYMM))

