
from megatron.objective import objective_one, objective
from computations.matrices.blas import SYRK, SYMM, GEMM
from sympy import MatrixSymbol, Symbol

n = Symbol('n')
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)

syrk = SYRK(1.0, A, 0.0, B)
gemm = GEMM(1.0, A.T, A, 0.0, B)
symm = SYMM(1.0, A.T, A, 0.0, B)

def test_objective_one():
    score = objective_one(syrk)
    assert isinstance(score, (int, float))

def test_objective_one_sorted():
    assert sorted((syrk, gemm, symm), key=objective_one) == [syrk, symm, gemm]

def test_objective():
    assert sorted((gemm+symm, gemm+syrk), key=objective) == [gemm + syrk, gemm + symm]
