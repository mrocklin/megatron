from sympy import MatrixSymbol, blockcut, block_collapse, assuming, Q
from megatron.core import compile
from computations.matrices.fortran.util import constant_arg
from megatron.util import remove


def test_blocked_symm():
    n = 1024
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    XX = blockcut(X, (n/2, n/2), (n/2, n/2))
    YY = blockcut(Y, (n/2, n/2), (n/2, n/2))
    with assuming(Q.symmetric(X)):
        expr = block_collapse(XX*YY)
        c = compile([X, Y], [expr])
    assert set(remove(constant_arg, c.inputs)) == set((X, Y))
    assert c.outputs[0].shape == (n, n)
