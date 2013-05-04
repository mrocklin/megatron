
from computations.matrices.blas import GEMM, SYMM, AXPY, SYRK
from computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from computations.matrices.fftw import FFTW

order = [FFTW, POSV, GESV, LASWP, SYRK, SYMM, GEMM, AXPY]

from computations.core import CompositeComputation

def objective_one(c):
    if type(c) in order:
        return order.index(type(c))
    else:
        return len(order) + 1

def objective(C):
    """ Sum of indices - not very robust """
    if isinstance(C, CompositeComputation):
        return sum(map(objective, C.computations))
    else:
        return objective_one(C)

