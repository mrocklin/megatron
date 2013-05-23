
identity = lambda x: x
def unique(seq, key=identity):
    seen = set()
    for item in seq:
        k = key(item)
        if k not in seen:
            seen.add(k)
            yield item

def intersect(a, b):
    return not not set(a).intersection(set(b))

def remove(predicate, collection):
    return [item for item in collection if not predicate(item)]

def merge(*dicts):
    """ Merge several dictionaries """
    out = dict()
    for d in dicts:
        out.update(d)
    return out


def groupby(f, coll):
    d = dict()
    for item in coll:
        key = f(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d

def chain(iters):
    for it in iters:
        for item in it:
            yield item
