from sympy import Q, assuming
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import make_kalman_comp
from megatron.scheduling.times import (make_compcost, make_commcost,
        make_commcost_tompkins)
import numpy as np

nn, nk = 1000, 1000

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
with assuming(*(assumptions+types)):
    compcost = make_compcost(c, inputs, ninputs)
commcost = make_commcost(7.058e-5, 6.87e-7)
commcost_tomp = make_commcost_tompkins(7.058e-5, 6.87e-7)


from computations.matrices.mpi import isend, irecv
from computations.matrices.blas import COPY
from computations.inplace import inplace_compile
from computations.matrices.io import disk_io
from computations.matrices.fortran.mpi import generate_mpi
from computations.core import CompositeComputation
from megatron.scheduling.util import make_order_cmp
from megatron.scheduling.tompkinslink import computation_from_dict

def make_heft_kalman():
    from heft import schedule as schedule_heft
    from heft import insert_sendrecvs, makespan

    with assuming(*(assumptions+types)):
        orders, jobson = schedule_heft(c.dict_io(), agents, compcost, commcost)
        neworders, jobson = insert_sendrecvs(orders, jobson, c.dict_io(),
                                             send=isend, recv=irecv)
        h0 = CompositeComputation(*[e.job for e in neworders[a]])
        h1 = CompositeComputation(*[e.job for e in neworders[b]])

        h0io= disk_io(h0, filenames)
        h1io= disk_io(h1, filenames)

        ih0 = inplace_compile(h0io, Copy=COPY)
        ih1 = inplace_compile(h1io, Copy=COPY)

        hcmpa = make_order_cmp([e.job for e in orders[a]])
        hcmpb = make_order_cmp([e.job for e in orders[b]])
        heftcode = generate_mpi(ih0, 'h0', [hcmpa], ih1, 'h1', [hcmpb])
    # Debug
    from computations.dot import writepdf
    writepdf(h0io, 'tmp/kalman_mpi_h0')
    writepdf(h1io, 'tmp/kalman_mpi_h1')
    with open('tmp/orders.txt', 'w') as f:
        f.write('\n'.join(map(str, orders.items())))

    return heftcode, makespan(orders)


def make_tompkins_kalman():
    from tompkins import schedule as schedule_tompkins
    from tompkins import orderings
    with assuming(*(assumptions+types)):
        commcost_tomp = make_commcost_tompkins(7.058e-5, 6.87e-7)
        dags, sched, m = schedule_tompkins(c.dict_io(), agents, compcost,
                commcost_tomp, send=isend, recv=irecv, M=2.0)
        torders = orderings(sched)
        t0 = computation_from_dict(dags[a])
        t1 = computation_from_dict(dags[b])

        t0io= disk_io(t0, filenames)
        t1io= disk_io(t1, filenames)
        it0 = inplace_compile(t0io, Copy=COPY)
        it1 = inplace_compile(t1io, Copy=COPY)

        tcmpa = make_order_cmp(torders[a])
        tcmpb = make_order_cmp(torders[b])
        tompcode = generate_mpi(it0, 't0', [tcmpa],
                                it1, 't1', [tcmpb])
    # Debug
    from computations.dot import writepdf
    writepdf(t0io, 'tmp/kalman_mpi_t0')
    writepdf(t1io, 'tmp/kalman_mpi_t1')
    with open('tmp/sched.txt', 'w') as f:
        f.write('\n'.join(map(str, sched)))

    return tompcode, m
