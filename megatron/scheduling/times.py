from heft import schedule, insert_sendrecvs
from computations.matrices.fortran.core import nbytes, build
from functools import partial

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
        return sum(map(partial(commtime, a=agent1, b=agent2), vars))
    return commcost

def make_commcost(*args):
    commtime = make_commtime(*args)
    def commcost(job1, job2, agent1, agent2):
        vars = set(job1.outputs) & set(job2.inputs)
        return sum(map(partial(commtime, a=agent1, b=agent2), vars))
    return commcost
commcost = make_commcost(inv_bandwidth, latency)

def profile_comp(comp):
    from computations.profile import ProfileMPI
    from computations.core import CompositeComputation
    return CompositeComputation(*map(ProfileMPI, comp.toposort()))

computations = lambda comp: comp.toposort()

def profile_build(pcomp, inputs, **kwargs):
    durations = [c.duration for c in computations(pcomp)]
    f = build(pcomp, inputs, durations, **kwargs)
    return f

def make_compcost(comp, inputs, ninputs, **kwargs):
    f = profile_build(profile_comp(comp), inputs, **kwargs)
    times = f(*ninputs)
    d = dict(zip(computations(comp), times))
    with open('tmp/compcost.dat', 'w') as f:
        for item in d.items():
            f.write("%s:   %s\n" % item)
    def compcost(job, agent):
        return d[job]
    return compcost