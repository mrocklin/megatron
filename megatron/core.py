import sympy.logpy
from sympy.logpy.core import asko
import computations.logpy.core
import computations.matrices.logpy.core

from megatron.patterns import patterns, vars

from logpy import Relation, facts
computes = Relation('computes')
facts(computes, *patterns)

from logpy import var, run, eq, variables
from logpy.assoccomm import eq_assoccomm as eqac
def computations_for(expr):
    c = var('comp')
    e = var('expr')
    pred = var('predicate')
    with variables(*vars):
        result = run(0, c, (computes, e, c, pred),
                           (eqac, e, expr),
                           (asko, pred, True))
    return result

from megatron.objective import objective
from computations.core import Identity
from computations.matrices.fortran.util import constant_arg
from megatron.util import remove
from sympy import assuming
def compile(inputs, outputs, *assumptions):
    """ A very simple greedy scheme.  Can walk into dead ends """
    c = Identity(*outputs)

    with assuming(*assumptions):
        while (set(remove(constant_arg, c.inputs)) != set(inputs)):
            possibilities = sum(map(computations_for, c.inputs), ())
            if not possibilities:     raise ValueError("Could not compile")
            best = min(possibilities, key=objective)
            c = c + best

    return c
