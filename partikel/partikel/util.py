import torch
import tree


def sum_structure(xs, dim=0, dim_closure=None):
    if dim_closure is not None:
        return tree.map_structure(lambda v: torch.sum(v, dim=dim_closure(v)), xs)
    else:
        return tree.map_structure(lambda v: torch.sum(v, dim=dim), xs)


def stack_structure(xs, dim=0):
    return tree.map_structure(lambda *v: torch.stack(v, dim=dim), *xs)
