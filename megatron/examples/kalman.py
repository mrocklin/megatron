from functools import partial
from sympy import Q, assuming
from computations.matrices.examples.kalman import (inputs, outputs, n, k,
        assumptions)
from computations.matrices.examples.kalman_comp import make_kalman_comp
from computations.profile import profile
from megatron.scheduling.times import (make_compcost, make_commcost,
        make_commcost_tompkins)
from computations.inplace import (inplace_compile, make_getname, tokenize,
        TokenComputation)
from computations.matrices.blas import COPY
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

tokenizer = make_getname()
ic = inplace_compile(c, tokenizer=tokenizer, Copy=COPY)
pcomp = profile(c)
ipcomp = profile(ic)


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
    compcost = make_compcost(ic, inputs, ninputs)
commcost = make_commcost(7.058e-5, 6.87e-7)
commcost_tomp = make_commcost_tompkins(7.058e-5, 6.87e-7)


from computations.matrices.mpi import isend, irecv
from computations.matrices.io import disk_io, ReadFromFile, WriteToFile
from computations.matrices.fortran.mpi import generate_mpi
from computations.core import CompositeComputation
from megatron.scheduling.util import make_order_cmp
from megatron.scheduling.tompkinslink import computation_from_dict

def make_heft_kalman():
    from heft import schedule as schedule_heft
    from heft import insert_sendrecvs, makespan

    send = partial(isend, tokenizer=tokenizer)
    recv = partial(irecv, tokenizer=tokenizer)

    with assuming(*(assumptions+types)):
        orders, jobson = schedule_heft(ic.dict_io(), agents, compcost, commcost)
        neworders, jobson = insert_sendrecvs(orders, jobson, c.dict_io(),
                                             send=send, recv=recv)
        h0 = CompositeComputation(*[e.job for e in neworders[a]])
        h1 = CompositeComputation(*[e.job for e in neworders[b]])

        reads0 = [wrap_tokenize(h0, ReadFromFile(filenames[i], i), tokenizer) for i in filenames if i in [v.expr for v in h0.inputs]]
        reads1 = [wrap_tokenize(h1, ReadFromFile(filenames[i], i), tokenizer) for i in filenames if i in [v.expr for v in h1.inputs]]
        writes0 = [wrap_tokenize(h0, WriteToFile(filenames[o], o), tokenizer) for o in filenames if o in [v.expr for v in h0.outputs]]
        writes1 = [wrap_tokenize(h1, WriteToFile(filenames[o], o), tokenizer) for o in filenames if o in [v.expr for v in h1.outputs]]
        disk0 = CompositeComputation(*(reads0 + writes0))
        disk1 = CompositeComputation(*(reads1 + writes1))

        h02 = h0 + disk0
        h12 = h1 + disk1

        hcmpa = make_order_cmp([e.job for e in orders[a]])
        hcmpb = make_order_cmp([e.job for e in orders[b]])
        heftcode = generate_mpi(h02, 'h0', [hcmpa], h12, 'h1', [hcmpb])
    # Debug
    from computations.dot import writepdf
    writepdf(h02, 'tmp/kalman_mpi_h0')
    writepdf(h12, 'tmp/kalman_mpi_h1')
    with open('tmp/orders.txt', 'w') as f:
        f.write('\n'.join(map(str, orders.items())))

    return heftcode, makespan(orders)


def wrap_tokenize(icomp, newcomp, tokenizer):
    known = {et.expr: et.token for et in icomp.inputs + icomp.outputs}
    input_tokens = [known[i] if i in known else tokenizer(i) for i in newcomp.inputs]
    output_tokens = [known[o] if o in known else tokenizer(o) for o in newcomp.outputs]
    return TokenComputation(newcomp, input_tokens, output_tokens)

def make_tompkins_kalman():
    from tompkins import schedule as schedule_tompkins
    from tompkins import orderings

    send = partial(isend, tokenizer=tokenizer)
    recv = partial(irecv, tokenizer=tokenizer)
    with assuming(*(assumptions+types)):
        commcost_tomp = make_commcost_tompkins(7.058e-5, 6.87e-7)
        dags, sched, m = schedule_tompkins(ic.dict_io(), agents, compcost,
                commcost_tomp, send=send, recv=recv, M=20.0)
        torders = orderings(sched)
        comps0 = set(dags[a].keys()) | set(sum(dags[a].values(), ()))
        comps1 = set(dags[b].keys()) | set(sum(dags[b].values(), ()))
        t0 = computation_from_dict(dags[a])
        t1 = computation_from_dict(dags[b])

        reads0 = [wrap_tokenize(t0, ReadFromFile(filenames[i], i), tokenizer) for i in filenames if i in [v.expr for v in t0.inputs]]
        reads1 = [wrap_tokenize(t1, ReadFromFile(filenames[i], i), tokenizer) for i in filenames if i in [v.expr for v in t1.inputs]]
        writes0 = [wrap_tokenize(t0, WriteToFile(filenames[o], o), tokenizer) for o in filenames if o in [v.expr for v in t0.outputs]]
        writes1 = [wrap_tokenize(t1, WriteToFile(filenames[o], o), tokenizer) for o in filenames if o in [v.expr for v in t1.outputs]]
        disk0 = CompositeComputation(*(reads0 + writes0))
        disk1 = CompositeComputation(*(reads1 + writes1))

        t02 = t0 + disk0
        t12 = t1 + disk1

        tcmpa = make_order_cmp(torders[a])
        tcmpb = make_order_cmp(torders[b])
        tompcode = generate_mpi(t02, 't0', [tcmpa],
                                t12, 't1', [tcmpb])
    # Debug
    from computations.dot import writepdf
    writepdf(t02, 'tmp/kalman_mpi_t0')
    writepdf(t12, 'tmp/kalman_mpi_t1')
    with open('tmp/sched.txt', 'w') as f:
        f.write('\n'.join(map(str, sched)))

    return tompcode, m

def write_kalman_data(n, k, directory='.'):
    ninputs = make_inputs(n, k)
    filenames = ['%s/%s.dat' % (directory, name) for name in map(str, inputs)]
    for ninput, filename in zip(ninputs, filenames):
        np.savetxt(filename, ninput, newline=' ')
