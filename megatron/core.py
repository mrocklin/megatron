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

def children(comp):
    atomics = sum(map(computations_for, comp.inputs), ())
    return map(comp.__add__, atomics)

from megatron.objective import objective
from computations.core import Identity
from computations.matrices.fortran.util import constant_arg
from megatron.util import remove
from sympy import assuming


import itertools as it
from functools import partial

def greedy(children, objective, isleaf, node):
    """ Greedy guided search in tree

    greedy      :: a -> a...    --  This function, produces lazy iterator
    children    :: a -> [a]     --  Children of node
    objective   :: a -> score   --  Quality of node
    isleaf      :: a -> T/F     --  Successful leaf of tree
    """
    if isleaf(node):
        return iter([node])
    f = partial(greedy, children, objective, isleaf)
    options = sorted(children(node), key=objective)
    streams = map(f, options)
    return it.chain(*streams)


def compile(inputs, outputs, *assumptions):
    """ A very simple greedy scheme.  Can walk into dead ends """
    c = Identity(*outputs)

    isleaf = lambda comp: set(remove(constant_arg, comp.inputs)) == set(inputs)

    with assuming(*assumptions):
        stream = greedy(children, objective, isleaf, c)
        result = next(stream)

    return result
