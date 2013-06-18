from computations.core import Computation, CompositeComputation
from computations.matrices import AXPY, COPY, isend, irecv

from sympy import MatrixSymbol, Symbol
n = Symbol('n')
A,B,C,X,Y,Z = [MatrixSymbol(a, n, n) for a in 'ABCXYZ']

a = AXPY(1, A, B)
b = AXPY(1, A+B, C)
c = AXPY(1, X, Y)
d = AXPY(1, X+Y, Z)
e = AXPY(1, A+B+C, X+Y+Z)

comp = a+b+c+d+e
agents = (0, 1)

compcost = lambda j, a: 1.0
commcost = lambda j, a, b: 0 if a==b else 1.5

from tompkins import schedule

dags, sched, makespan = schedule(comp.dict_io(), agents, compcost, commcost,
        send=isend, recv=irecv)

jobson = {s[0]: s[2] for s in sched}
times  = {s[0]: s[1] for s in sched}

def test_makespan():
    assert makespan == 4.5

def test_sched():
    assert jobson[a] != jobson[c]
    assert jobson[b] != jobson[d]

    assert times[a] == times[c] == 0
    assert 1.0 in (times[b], times[d])
