
from computations.matrices.blas import GEMM, SYMM, AXPY, SYRK
from computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from computations.matrices.fftw import FFTW
from sympy.matrices.expressions.fourier import DFT
from sympy.matrices.expressions import Transpose, Inverse, MatMul, MatAdd

order = [FFTW, POSV, GESV, LASWP, SYRK, SYMM, GEMM, AXPY]

from computations.core import CompositeComputation

def objective_one(c):
    if type(c) in order:
        return order.index(type(c))
    else:
        return len(order) + 1

def objective(C):
    """ Sum of indices - not very robust """
    if not valid(C):
        return 1e9
    if isinstance(C, CompositeComputation):
        return sum(map(objective, C.computations))
    else:
        return objective_one(C)

def invalid_input(i):
    return (isinstance(i, DFT) or
            isinstance(i, Transpose) and isinvalid_input(i.arg) or
            isinstance(i, Inverse) or
            # isinstance(i, MatMul) and isinstance(i.args[-1], Inverse) or
            isinstance(i, MatMul) and invalid_input(i.args[-1]) or
            isinstance(i, MatAdd) and any(invalid_input(arg) for arg in i.args)
            )


def valid(C):
    return not any(invalid_input(i) for i in C.inputs)
