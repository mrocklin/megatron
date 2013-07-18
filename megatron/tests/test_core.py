from sympy import Symbol, MatrixSymbol, ask, Q, assuming, ZeroMatrix
from computations.matrices.blas import SYRK, GEMM, SYMM
from megatron.core import computations_for, compile, vars, variables

n = Symbol('n')
X = MatrixSymbol('X', n, n)
expr = X * X.T

def test_correct_computation_matches():
    with variables(*vars):
        assert set(map(type, computations_for(expr))) == set((SYRK, GEMM))

        with assuming(Q.symmetric(X)):
            assert set(map(type, computations_for(expr))) == set((SYRK, GEMM, SYMM))

def test_computations_reify_correctly():
    print computations_for(expr)
    assert all(X in c.inputs for c in computations_for(expr))

def test_compile():
    assert str(compile([X], [X*X.T])) == str(SYRK(1.0, X, 0, ZeroMatrix(n, n)))

def test_gesv_laswp():
    from computations.matrices.lapack import (IPIV, PermutationMatrix, LASWP,
            GESV)
    X = MatrixSymbol('X', 3, 3)
    Z = MatrixSymbol('Z', 3, 3)
    comp = GESV(Z, X) +LASWP(PermutationMatrix(IPIV(Z.I*X))*Z.I*X, IPIV(Z.I*X))
    assert Z.I*X in comp.outputs
    assert 'IPIV' not in str(comp.outputs)
