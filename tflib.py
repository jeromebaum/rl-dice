import tensorflow as tf


def flatten_unflatten(variable):
    """
    >>> apply_snd_to_fst = lambda xf: xf[1](xf[0])
    >>> fu_identity = lambda x: apply_snd_to_fst(flatten_unflatten(x)) == x
    >>> flatten_unflatten('foo')[0]
    ['foo']
    >>> x = {'foo': ([1,(5.,)],)}
    >>> flatten_unflatten(x)[0]
    [1, 5.0]
    >>> fu_identity(x)
    True
    """
    def flatten_unflatten_raw(variable):
        if type(variable) is tuple:
            flattened, unflatten_list = flatten_unflatten_raw(list(variable))
            def unflatten_raw(l):
                result, rest = unflatten_list(l)
                return tuple(result), rest
        elif type(variable) is list:
            flattened = sum([flatten_unflatten_raw(v)[0] for v in variable], [])
            def unflatten_raw(l):
                rest = l
                result = []
                for unflattener in [flatten_unflatten_raw(v)[1] for v in variable]:
                    v, rest = unflattener(rest)
                    result += [v]
                return result, rest
        elif type(variable) is dict:
            flattened = sum([flatten_unflatten_raw(v)[0] for v in variable.values()], [])
            def unflatten_raw(l):
                rest = l
                result = {}
                for k, unflattener in [(k, flatten_unflatten_raw(v)[1]) for (k, v) in variable.items()]:
                    v, rest = unflattener(rest)
                    result[k] = v
                return result, rest
        else:
            def unflatten_raw(l):
                return l[0], l[1:]
            return [variable], unflatten_raw
        return flattened, unflatten_raw
    flattened, unflatten_raw = flatten_unflatten_raw(variable)
    def unflatten(l):
        unflattened, [] = unflatten_raw(l)
        return unflattened
    return flattened, unflatten


def complex_tuple(variable, *args, **kwargs):
    flattened, unflatten = flatten_unflatten(variable)
    return unflatten(tf.tuple(flattened, *args, **kwargs))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
