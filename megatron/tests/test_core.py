from megatron.core import computations_for, compile

from sympy import Symbol, MatrixSymbol, ask, Q, assuming, ZeroMatrix


n = Symbol('n')
X = MatrixSymbol('X', n, n)

from computations.matrices.blas import SYRK, GEMM, SYMM
expr = X * X.T

def test_correct_computation_matches():
    assert set(map(type, computations_for(expr))) == set((SYRK, GEMM))

    with assuming(Q.symmetric(X)):
        assert set(map(type, computations_for(expr))) == set((SYRK, GEMM, SYMM))

def test_computations_reify_correctly():
    print computations_for(expr)
    assert all(X in c.inputs for c in computations_for(expr))

def test_compile():
    from computations.matrices.blas import SYRK
    assert compile([X], [X*X.T]) == SYRK(1.0, X, 0, ZeroMatrix(n, n))
