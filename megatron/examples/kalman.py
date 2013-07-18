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

commcost = make_commcost(7.058e-5, 6.87e-7)
commcost_tomp = make_commcost_tompkins(7.058e-5, 6.87e-7)


from computations.matrices.mpi import isend, irecv
from computations.matrices.io import disk_io, ReadFromFile, WriteToFile
from computations.matrices.fortran.mpi import generate_mpi
from computations.core import CompositeComputation
from megatron.scheduling.util import make_order_cmp
from megatron.scheduling.tompkinslink import computation_from_dict

def make_heft_kalman():
    return make_heft(c, (a, b), types+assumptions, commcost, inputs, ninputs)

def make_tompkins_kalman():
    return make_tompkins(c, (a, b), types+assumptions, commcost_tomp, inputs, ninputs)

from heft import schedule as schedule_heft
from heft import insert_sendrecvs, makespan
def make_heft(c, agents, assumptions, commcost, inputs, ninputs):
    tokenizer = make_getname()
    ic = inplace_compile(c, tokenizer=tokenizer, Copy=COPY)

    send = partial(isend, tokenizer=tokenizer)
    recv = partial(irecv, tokenizer=tokenizer)

    with assuming(*assumptions):
        compcost = make_compcost(ic, inputs, ninputs)

        orders, jobson = schedule_heft(ic.dict_io(), agents, compcost, commcost)
        neworders, jobson = insert_sendrecvs(orders, jobson, c.dict_io(),
                                             send=send, recv=recv)

        args = []
        for ii, a in enumerate(agents):
            h = CompositeComputation(*[e.job for e in neworders[a]])

            reads = [wrap_tokenize(h, ReadFromFile(filenames[i], i), tokenizer)
                      for i in filenames if i in [v.expr for v in h.inputs]]
            writes = [wrap_tokenize(h, WriteToFile(filenames[o], o), tokenizer)
                      for o in filenames if o in [v.expr for v in h.outputs]]
            disk = CompositeComputation(*(reads + writes))

            h2 = h + disk

            cmp = make_order_cmp([e.job for e in orders[a]])

            args.extend([h2, 'f%d'%ii, [cmp]])

            # Debug
            from computations.dot import writepdf
            writepdf(h2, 'tmp/prog_mpi_h%d'%ii)

        heftcode = generate_mpi(*args)

    # Debug
    with open('tmp/orders.txt', 'w') as f:
        f.write('\n'.join(map(str, orders.items())))

    return heftcode, makespan(orders)



from tompkins import schedule as schedule_tompkins
from tompkins import orderings
def make_tompkins(c, agents, assumptions, commcost, inputs, ninputs):
    tokenizer = make_getname()
    ic = inplace_compile(c, tokenizer=tokenizer, Copy=COPY)

    send = partial(isend, tokenizer=tokenizer)
    recv = partial(irecv, tokenizer=tokenizer)

    with assuming(*assumptions):
        compcost = make_compcost(ic, inputs, ninputs)

        dags, sched, m = schedule_tompkins(ic.dict_io(), agents, compcost,
                commcost, send=send, recv=recv, M=20.0)
        torders = orderings(sched)

        args = []

        for ii, a in enumerate(agents):
            comps = set(dags[a].keys()) | set(sum(dags[a].values(), ()))
            t = computation_from_dict(dags[a])

            reads = [wrap_tokenize(t, ReadFromFile(filenames[i], i), tokenizer) for i in filenames if i in [v.expr for v in t.inputs]]
            writes = [wrap_tokenize(t, WriteToFile(filenames[o], o), tokenizer) for o in filenames if o in [v.expr for v in t.outputs]]
            disk = CompositeComputation(*(reads + writes))

            t2 = t + disk

            cmp = make_order_cmp(torders[a])

            args.extend([t2, 'f%d'%ii, [cmp]])

            # Debug
            from computations.dot import writepdf
            writepdf(t2, 'tmp/prog_mpi_t%d'%ii)

        tompcode = generate_mpi(*args)

    # Debug
    with open('tmp/sched.txt', 'w') as f:
        f.write('\n'.join(map(str, sched)))

    return tompcode, m

def write_kalman_data(n, k, directory='.'):
    ninputs = make_inputs(n, k)
    filenames = ['%s/%s.dat' % (directory, name) for name in map(str, inputs)]
    for ninput, filename in zip(ninputs, filenames):
        np.savetxt(filename, ninput, newline=' ')

def wrap_tokenize(icomp, newcomp, tokenizer):
    known = {et.expr: et.token for et in icomp.inputs + icomp.outputs}
    input_tokens = [known[i] if i in known else tokenizer(i) for i in newcomp.inputs]
    output_tokens = [known[o] if o in known else tokenizer(o) for o in newcomp.outputs]
    return TokenComputation(newcomp, input_tokens, output_tokens)

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
