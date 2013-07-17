import sympy.logpy
from sympy.logpy.core import asko
import computations.logpy.core
import computations.matrices.logpy.core
from computations.core import CompositeComputation

from megatron.patterns import patterns, vars

from logpy import Relation, facts
computes = Relation('computes')
facts(computes, *patterns)

from logpy import var, run, eq, variables
from logpy.assoccomm import eq_assoccomm as eqac
def computations_for(expr):
    """ Computations that can break down expr """
    c = var('comp')
    e = var('expr')
    pred = var('predicate')
    result = run(0, c, (computes, e, c, pred),
                       (eqac, e, expr),
                       (asko, pred, True))
    return result

def children(comp):
    """ Compute next options in tree of possible algorithms """
    atomics = it.chain.from_iterable(it.imap(computations_for, comp.inputs))
    atomics = (a for a in atomics if a.typecheck() and valid(a))
    return it.imap(comp.__add__, atomics)

from megatron.objective import objective, valid
from computations.core import Identity
from computations.matrices.fortran.util import constant_arg
from megatron.util import remove, chain
from sympy import assuming


import itertools as it
from functools import partial

debug = False

def blind(children, objective, isleaf, node):
    """ Blind search in tree

    greedy      :: a -> a...    --  This function, produces lazy iterator
    children    :: a -> [a]     --  Children of node
    objective   :: a -> score   --  Quality of node (not used)
    isleaf      :: a -> T/F     --  Successful leaf of tree
    """
    if debug:
        print "Node:   ", node, '\n\n'
        print "Inputs: ", '\n'.join(map(str,node.inputs)), '\n\n'
    if isleaf(node):
        if debug:
            print "Is Leaf"
        return iter([node])
    return blind(children, objective, isleaf, next(children(node)))

def greedy(children, objective, isleaf, node):
    """ Greedy guided search in tree

    greedy      :: a -> a...    --  This function, produces lazy iterator
    children    :: a -> [a]     --  Children of node
    objective   :: a -> score   --  Quality of node
    isleaf      :: a -> T/F     --  Successful leaf of tree
    """
    if debug:
        print "Node:   ", node, '\n\n'
        print "Inputs: ", '\n'.join(map(str,node.inputs)), '\n\n'
    if isleaf(node):
        if debug:
            print "Is Leaf"
        return iter([node])
    f = partial(greedy, children, objective, isleaf)
    options = sorted(children(node), key=objective)
    if debug:
        from computations.dot import show
        map(show, options)
    streams = it.imap(f, options)
    return it.chain.from_iterable(streams)


def compile(inputs, outputs, *assumptions, **kwargs):
    """ A very simple greedy scheme."""

    strat = kwargs.get('strat', greedy)
    c = Identity(*outputs)

    # Is this computation a leaf in our tree?  Do its inputs match ours?
    isleaf = lambda comp: set(remove(constant_arg, comp.inputs)) == set(inputs)

    with assuming(*assumptions):
        with variables(*vars):
            stream = strat(children, objective, isleaf, c) # all valid computations
            result = next(stream)                           # first valid computtion

    return result
