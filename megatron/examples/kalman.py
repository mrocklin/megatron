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

a, b = agents = (0, 1)

comp = make_kalman_comp(*inputs)

latency, bandwidth = 2.7e-4, 1.1e8
invbandwidth = 1./bandwidth
commcost = make_commcost(latency, invbandwidth)
commcost_tomp = make_commcost_tompkins(latency, invbandwidth)

def make_inputs(n, k):
    nmu = np.random.rand(n)
    nsigma = np.random.rand(n, n); nsigma = np.dot(nsigma, nsigma.T)
    nh = np.random.rand(k, n)
    nr = np.random.rand(k, k); nr = np.dot(nr, nr.T)
    ndata = np.random.rand(k)

    ninputs = nmu, nsigma, nh, nr, ndata

    ninputs = map(np.asfortranarray, ninputs)  # force fortran ordering
    return ninputs

ninputs = make_inputs(nn, nk)

filenames = {mu: 'mu.dat', Sigma: 'Sigma.dat', H: 'H.dat', R: 'R.dat',
             data: 'data.dat', newmu: 'mu2.dat', newSigma: 'Sigma2.dat'}
filenames = {k: 'tmp/'+v for k,v in filenames.items()}
for input, ninput in zip(inputs, ninputs):
    np.savetxt(filenames[input], ninput, newline=' ')

def make_heft_kalman():
    ninputs = make_inputs(nn, nk)
    return make_heft(comp, (a, b), types+assumptions, commcost, inputs, ninputs,
            filenames)

def make_tompkins_kalman():
    ninputs = make_inputs(nn, nk)
    return make_tompkins(comp, (a, b), types+assumptions, commcost_tomp, inputs,
            ninputs, filenames)
