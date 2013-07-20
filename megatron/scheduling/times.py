from heft import schedule, insert_sendrecvs
from computations.matrices.fortran.core import nbytes, build
from functools import partial
from computations.inplace import inplace_compile, ExprToken
from computations.matrices.blas import COPY

inv_bandwidth = 6.87213759e-07
latency = 7.05801143e-05

def make_commtime(latency, inv_bandwidth):
    def commtime(var, a, b):
        if a == b:
            return 0
        return nbytes(var) * inv_bandwidth + latency
    return commtime
commtime = make_commtime(inv_bandwidth, latency)

def make_commcost_tompkins(*args):
    commtime = make_commtime(*args)
    def commcost(job, agent1, agent2):
        vars = job.outputs
        vars = [var.expr if isinstance(var, ExprToken) else var
                for var in vars]
        return sum(map(partial(commtime, a=agent1, b=agent2), vars))
    return commcost

def make_commcost(*args):
    commtime = make_commtime(*args)
    def commcost(job1, job2, agent1, agent2):
        vars = set(job1.outputs) & set(job2.inputs)
        vars = [var.expr if isinstance(var, ExprToken) else var
                for var in vars]
        return sum(map(partial(commtime, a=agent1, b=agent2), vars))
    return commcost
commcost = make_commcost(inv_bandwidth, latency)

from computations.profile import profile
computations = lambda comp: comp.toposort()

def profile_build(pcomp, inputs, **kwargs):
    durations = [c.duration for c in computations(pcomp)]
    key = str(abs(hash(pcomp)))
    filename = kwargs.pop('filename', 'profile_%s.f90'%key)
    modname  = kwargs.pop('modname', 'profile_%s'%key)
    f = build(pcomp, inputs, durations, modname=modname,
                                        filename=filename,
                                        **kwargs)
    return f

def make_compcost(comp, inputs, ninputs, **kwargs):
    f = profile_build(profile(comp), inputs, **kwargs)
    times = f(*ninputs)
    assert all(isinstance(time, float) for time in times)
    d = dict(zip(computations(comp), times))
    with open('tmp/compcost.dat', 'w') as f:
        for item in d.items():
            f.write("%s:   %s\n" % item)
    def compcost(job, agent):
        return d[job]
    return compcost
