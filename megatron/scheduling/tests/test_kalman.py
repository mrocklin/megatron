from sympy import Q, assuming
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import make_kalman_comp
from megatron.scheduling.times import make_compcost, commcost

import numpy as np

nn, nk = 1000, 500

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

def compcost_kalman(nn=1000, nk=500):
    ninputs = make_inputs(nn, nk)

    types = tuple(map(Q.real_elements, inputs))
    with assuming(*(assumptions+types)):
        compcost = make_compcost(c, inputs, ninputs)

    return compcost

def test_compcost():
    compcost = compcost_kalman()
    assert all(isinstance(compcost(comp, 1), float) for comp in c.computations)

def test_integrative():
    from heft import schedule, insert_sendrecvs
    from computations.matrices.mpi import isend, irecv
    from computations.matrices.blas import COPY
    from computations.inplace import inplace_compile
    from computations.matrices.io import disk_io
    from computations.matrices.fortran.mpi import generate_mpi
    from computations.core import CompositeComputation
    ninputs = make_inputs(nn, nk)
    types = tuple(map(Q.real_elements, inputs))
    mu, Sigma, H, R, data = inputs
    newmu, newSigma = outputs
    filenames = {mu: 'mu.dat', Sigma: 'Sigma.dat', H: 'H.dat', R: 'R.dat',
                 data: 'data.dat', newmu: 'mu2.dat', newSigma: 'Sigma2.dat'}

    with assuming(*(assumptions+types)):
        compcost = make_compcost(c, inputs, ninputs)
        orders, jobson = schedule(c.dict_oi(), (0,1), compcost, commcost)
        neworders, jobson = insert_sendrecvs(orders, jobson, c.dict_io(),
                                             send=isend, recv=irecv)
        c1 = CompositeComputation(*[e.job for e in neworders[0]])
        c2 = CompositeComputation(*[e.job for e in neworders[1]])
        c1io = disk_io(c1, filenames)
        c2io = disk_io(c2, filenames)
        ic1io = inplace_compile(c1io, Copy=COPY)
        ic2io = inplace_compile(c2io, Copy=COPY)
        code = generate_mpi(ic1io, [],  [], 'c1', ic2io, [], [], 'c2')
        with open('tmp/kalman_mpi.f90', 'w') as f:
            f.write(code)
        assert isinstance(code, str)

    from computations.dot import writepdf
    writepdf(c1io, 'tmp/kalman_mpi_1')
    writepdf(c2io, 'tmp/kalman_mpi_2')

def write_kalman_data(n, k, directory='.'):
    ninputs = make_inputs(n, k)
    filenames = ['%s/%s.dat' % (directory, name) for name in map(str, inputs)]
    for ninput, filename in zip(ninputs, filenames):
        np.savetxt(filename, ninput, newline=' ')
