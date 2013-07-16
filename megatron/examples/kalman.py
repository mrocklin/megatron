
from sympy import Q, assuming
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import make_kalman_comp
from megatron.scheduling.times import make_compcost
import numpy as np

nn, nk = 1000, 1000

dimsubs = lambda x: x.subs({n: nn, k: nk})
inputs = map(dimsubs, inputs)
outputs = map(dimsubs, outputs)
assumptions = tuple(map(dimsubs, assumptions))

c = make_kalman_comp(*inputs)

def make_inputs(n, k):
    ninputs = [
        np.ones(n, dtype=np.float64),
        np.eye(n, dtype=np.float64),
        np.hstack([np.eye(k, dtype=np.float64),
                              np.zeros((k, n-k), dtype=np.float64)]),
        np.eye(k, dtype=np.float64),
        np.ones(k, dtype=np.float64)
    ]

    ninputs = map(np.asfortranarray, ninputs)  # force fortran ordering
    return ninputs

def compcost_kalman(nn=nn, nk=nk):
    ninputs = make_inputs(nn, nk)

    types = tuple(map(Q.real_elements, inputs))
    with assuming(*(assumptions+types)):
        compcost = make_compcost(c, inputs, ninputs)

    return compcost

