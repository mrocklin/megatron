from computations.core import CompositeComputation

def computation_from_dict(d):
    """ Transform at dict-dag to a CompositeComputation

    dict-dag like    : {a: (b, c), b: (), c: ()}
    computation like : CompositeComputation(a, b, c)
    """
    return CompositeComputation(*
            set(d.keys()) | set(sum(d.values(), ())))
