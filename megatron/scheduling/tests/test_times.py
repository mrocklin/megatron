from sympy import MatrixSymbol, ZeroMatrix, Q, assuming, Number
from computations.matrices.blas import GEMM
from computations.matrices.lapack import GESV
from megatron.scheduling.times import make_commcost, make_compcost
import numpy as np

n = 2
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
x = MatrixSymbol('x', n, 1)


gemm = GEMM(1, A, B, 0, ZeroMatrix(n, n))
gemm2 = GEMM(1, A*B, x, 0, ZeroMatrix(n, 1))

def test_commcost():
    latency = 1.
    inv_bandwidth = .1
    commcost = make_commcost(latency, inv_bandwidth)
    with assuming(*map(Q.real_elements, [A, B, x])):
        time = commcost(gemm, gemm2, 1, 2)
        assert isinstance(time, (float, Number))
        assert time == 8*n*n*inv_bandwidth + latency
        assert commcost(gemm, gemm2, 1, 1) == 0

def test_compcost():
    nA = np.asarray([[1, 2], [3, 4]], dtype=np.float64, order='F')
    nB = np.asarray([[1, 0], [0, 1]], dtype=np.float64, order='F')
    nx = np.asarray([[1], [1]], dtype=np.float64, order='F')

    with assuming(*map(Q.real_elements, [A, B, x])):
        compcost = make_compcost(gemm + gemm2, [A, B, x], [nA, nB, nx],
                filename='compcost.f90', modname='compcost')
    time = compcost(gemm, 1)
    assert isinstance(time, (float, Number))
    assert 0 < time < .1
