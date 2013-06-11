from sympy import Q, assuming
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import make_kalman_comp
from megatron.scheduling.times import make_compcost

import numpy as np

nn, nk = 1000, 500

dimsubs = lambda x: x.subs({n: nn, k: nk})
inputs = map(dimsubs, inputs)
outputs = map(dimsubs, outputs)
assumptions = tuple(map(dimsubs, assumptions))

c = make_kalman_comp(*inputs)

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

def compcost_kalman(nn=1000, nk=500):
    ninputs = make_inputs(nn, nk)

    types = tuple(map(Q.real_elements, inputs))
    with assuming(*(assumptions+types)):
        compcost = make_compcost(c, inputs, ninputs)

    return compcost

def test_compcost():
    compcost = compcost_kalman()
    assert all(isinstance(compcost(comp, 1), float) for comp in c.computations)
