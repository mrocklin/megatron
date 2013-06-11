from sympy import Q, assuming
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import c
from megatron.scheduling.times import make_compcost

import numpy as np

def make_inputs(nn, nk):
    dims = {n: nn, k: nk}

    # random numerical inputs
    ninputs = [np.random.rand(*i.subs(dims).shape) for i in inputs]

    # Sigma and R must be symmetric positive definite.  Just set as identity
    for i in [1, 3]:
        ninputs[i] = np.eye(ninputs[i].shape[0])

    # force fortran ordering
    ninputs = map(np.asfortranarray, ninputs)
    return ninputs

def test_kalman():
    ninputs = make_inputs(100, 100)

    types = tuple(map(Q.real_elements, inputs))
    with assuming(*(assumptions+types)):
        compcost = make_compcost(c, inputs, ninputs)

    assert all(isinstance(compcost(comp, 1), float) for comp in c.computations)
