
def make_order_cmp(order):
    """

    >>> from megatron.scheduling.util import make_order_cmp
    >>> order = [1, 5, 2, 3, 0]
    >>> cmp = make_order_cmp(order)
    >>> cmp(5, 3)
    -1
    >>> cmp(3, 5)
    1
    >>> cmp(5, 10000)
    0
    """
    def order_cmp(a, b):
        if a in order and b in order:
            return order.index(b) - order.index(a)
        else:
            return 0
    return order_cmp
