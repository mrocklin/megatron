from megatron.scheduling.times import make_commcost, make_commcost_tompkins
from megatron.scheduling.core import make_heft, make_tompkins

from computations.profile import profile
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import make_kalman_comp

from sympy import Q

import numpy as np

nn, nk = 2000, 1000

dimsubs = lambda x: x.subs({n: nn, k: nk})
inputs = map(dimsubs, inputs)
outputs = map(dimsubs, outputs)

mu, Sigma, H, R, data = inputs
newmu, newSigma = outputs

assumptions = tuple(map(dimsubs, assumptions))
types = tuple(map(Q.real_elements, inputs))

filenames = {mu: 'mu.dat', Sigma: 'Sigma.dat', H: 'H.dat', R: 'R.dat',
             data: 'data.dat', newmu: 'mu2.dat', newSigma: 'Sigma2.dat'}

a, b = agents = (0, 1)

c = make_kalman_comp(*inputs)

latency, bandwidth = 2.7e-4, 1.1e8
invbandwidth = 1./bandwidth
commcost = make_commcost(latency, invbandwidth)
commcost_tomp = make_commcost_tompkins(latency, invbandwidth)

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

ninputs = make_inputs(nn, nk)

def make_heft_kalman():
    return make_heft(c, (a, b), types+assumptions, commcost, inputs, ninputs,
            filenames)

def make_tompkins_kalman():
    return make_tompkins(c, (a, b), types+assumptions, commcost_tomp, inputs,
            ninputs, filenames)

def write_kalman_data(n, k, directory='.'):
    ninputs = make_inputs(n, k)
    filenames_ = {k: "%s/%s"%(directory, v) for k, v in filenames.items()}
    for input, ninput in zip(inputs, ninputs):
        np.savetxt(filenames_[input], ninput, newline=' ')
