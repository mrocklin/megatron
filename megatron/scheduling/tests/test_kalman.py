from megatron.examples.kalman import make_heft_kalman, make_tompkins_kalman

def test_integrative():
    heftcode, makespan_heft = make_heft_kalman()
    assert isinstance(heftcode, str)

    tompcode, makespan_tompkins = make_tompkins_kalman()
    assert isinstance(tompcode, str)


    print "Heft     Makespan: ", makespan_heft
    print "Tompkins Makespan: ", makespan_tompkins
    assert makespan_tompkins <= makespan_heft
    assert 2*makespan_tompkins >= makespan_heft
