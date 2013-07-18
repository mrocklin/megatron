from sympy import Q, assuming
from megatron.examples.kalman import (inputs, outputs, n, k,
        assumptions, c, nn, nk, make_inputs, make_heft_kalman,
        make_tompkins_kalman)
from megatron.scheduling.times import (make_compcost, make_commcost,
        make_commcost_tompkins)

import numpy as np


def test_integrative():
    heftcode, makespan_heft = make_heft_kalman()
    assert isinstance(heftcode, str)

    tompcode, makespan_tompkins = make_tompkins_kalman()
    assert isinstance(tompcode, str)


    print "Heft     Makespan: ", makespan_heft
    print "Tompkins Makespan: ", makespan_tompkins
    assert makespan_tompkins <= makespan_heft
    assert 2*makespan_tompkins >= makespan_heft
