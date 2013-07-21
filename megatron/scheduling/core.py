from functools import partial
from computations.matrices.mpi import isend, irecv
from computations.matrices.io import ReadFromFile, WriteToFile
from computations.matrices.fortran.mpi import generate_mpi
from computations.matrices.blas import COPY
from computations.core import CompositeComputation
from computations.inplace import (inplace_compile, make_getname, tokenize,
        TokenComputation)
from megatron.scheduling.util import make_order_cmp, wrap_tokenize
from megatron.scheduling.tompkinslink import computation_from_dict
from megatron.scheduling.times import (make_compcost, make_commcost,
        make_commcost_tompkins)

from sympy import Q, assuming
import numpy as np

from heft import schedule as schedule_heft
from heft import insert_sendrecvs, makespan

def make_heft(c, agents, assumptions, commcost, inputs, ninputs, filenames):
    tokenizer = make_getname()
    ic = inplace_compile(c, tokenizer=tokenizer, Copy=COPY)

    send = partial(isend, tokenizer=tokenizer)
    recv = partial(irecv, tokenizer=tokenizer)

    with assuming(*assumptions):
        compcost = make_compcost(ic, inputs, ninputs)

        orders, jobson = schedule_heft(ic.dict_io(), agents, compcost, commcost)
        neworders, jobson = insert_sendrecvs(orders, jobson, ic.dict_io(),
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
def make_tompkins(c, agents, assumptions, commcost, inputs, ninputs, filenames,
        **kwargs):
    tokenizer = make_getname()
    ic = inplace_compile(c, tokenizer=tokenizer, Copy=COPY)

    send = partial(isend, tokenizer=tokenizer)
    recv = partial(irecv, tokenizer=tokenizer)

    with assuming(*assumptions):
        compcost = make_compcost(ic, inputs, ninputs)

        M = kwargs.pop('M', 20.0)
        dags, sched, m = schedule_tompkins(ic.dict_io(), agents, compcost,
                commcost, send=send, recv=recv, M=M)
        torders = orderings(sched)

        args = []

        for ii, a in enumerate(dags):
            comps = set(dags[a].keys()) | set(sum(dags[a].values(), ()))
            t = computation_from_dict(dags[a])

            reads = [wrap_tokenize(t, ReadFromFile(filenames[i], i), tokenizer)
                        for i in filenames if i in [v.expr for v in t.inputs]]
            writes = [wrap_tokenize(t, WriteToFile(filenames[o], o), tokenizer)
                        for o in filenames if o in [v.expr for v in t.outputs]]
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

def write_inputs(inputs, ninputs, directory='.'):
    filenames = ['%s/%s.dat' % (directory, name) for name in map(str, inputs)]
    for ninput, filename in zip(ninputs, filenames):
        np.savetxt(filename, ninput, newline=' ')
