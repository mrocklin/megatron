import sympy.logpy
from sympy.logpy.core import asko
import computations.logpy.core
import computations.matrices.logpy.core

from megatron.patterns import patterns, vars

from logpy import Relation, facts
computes = Relation('computes')
facts(computes, *patterns)

from logpy import var, run, eq
from logpy.variables import variables
def computations_for(expr):
    c = var('comp')
    e = var('expr')
    pred = var('predicate')
    with variables(*vars):
        result = run(0, c, (computes, e, c, pred),
                           (eq, e, expr),
                           (asko, pred, True))
    return result

from megatron.objective import objective
from computations.core import Identity
from computations.matrices.fortran.util import constant_arg
from megatron.util import remove
def compile(outputs, inputs):
    c = Identity(*outputs)

    while (set(remove(constant_arg, c.inputs)) != set(inputs)):
        c = c + min(sum(map(computations_for, c.inputs), ()), key=objective)

    return c
