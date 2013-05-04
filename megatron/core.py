import sympy.logpy
from sympy.logpy.core import asko
import computations.logpy


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
        result = run(0, c, computes(e, c, pred),
                           (eq, e, expr),
                           (asko, pred, True))
    return result
