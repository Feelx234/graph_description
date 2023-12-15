from typing import Optional

import numpy as np
Tensor = type(np.array(0))
from collections import namedtuple
Data = namedtuple("Data", ['x', 'y', 'edge_index', 'train_mask', 'val_mask', 'test_mask'], defaults=(None, None, None))

def mask_select(src: Tensor, dim: int, mask: Tensor) -> Tensor:
    r"""Returns a new tensor which masks the :obj:`src` tensor along the
    dimension :obj:`dim` according to the boolean mask :obj:`mask`.

    Args:
        src (torch.Tensor): The input tensor.
        dim (int): The dimension in which to mask.
        mask (torch.BoolTensor): The 1-D tensor containing the binary mask to
            index with.
    """
    assert mask.dim() == 1

    if not torch.jit.is_scripting():
        if isinstance(src, TensorFrame):
            assert dim == 0 and src.num_rows == mask.numel()
            return src[mask]

    assert src.size(dim) == mask.numel()
    dim = dim + src.dim() if dim < 0 else dim
    assert dim >= 0 and dim < src.dim()

    # Applying a 1-dimensional mask in the first dimension is significantly
    # faster than broadcasting the mask and utilizing `masked_select`.
    # As such, we transpose in the first dimension, perform the masking, and
    # then transpose back to the original shape.
    src = src.transpose(0, dim) if dim != 0 else src
    out = src[mask]
    out = out.transpose(0, dim) if dim != 0 else out

    return out


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        index (Tensor): The indices.
        size (int, optional): The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:
        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.ravel()
    size = int(index.max()) + 1 if size is None else size
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:
        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple=False).view(-1)



import typing
from typing import List, Optional, Tuple, Union

#from torch_geometric.utils import index_sort, scatter
#from torch_geometric.utils.num_nodes import maybe_num_nodes


MISSING = '???'

def index_sort(
    inputs: Tensor,
    max_value: Optional[int] = None,
    stable: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Sorts the elements of the :obj:`inputs` tensor in ascending order.
    It is expected that :obj:`inputs` is one-dimensional and that it only
    contains positive integer values. If :obj:`max_value` is given, it can
    be used by the underlying algorithm for better performance.

    Args:
        inputs (torch.Tensor): A vector with positive integer values.
        max_value (int, optional): The maximum value stored inside
            :obj:`inputs`. This value can be an estimation, but needs to be
            greater than or equal to the real maximum.
            (default: :obj:`None`)
        stable (bool, optional): Makes the sorting routine stable, which
            guarantees that the order of equivalent elements is preserved.
            (default: :obj:`False`)
    """
    order = np.argsort(inputs)
    return inputs[order], order

def maybe_num_nodes(edge_index, num_nodes=None):
    if not num_nodes is None:
        return num_nodes
    else:
        return edge_index.ravel().max()+1


def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Tensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Example:
        >>> edge_index = torch.tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = torch.tensor([1., 1., 1., 1.])
        >>> coalesce(edge_index)
        tensor([[1, 2, 3],
                [3, 1, 2]])

        >>> # Sort `edge_index` column-wise
        >>> coalesce(edge_index, sort_by_row=False)
        tensor([[2, 3, 1],
                [1, 2, 3]])

        >>> coalesce(edge_index, edge_attr)
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>> coalesce(edge_index, edge_attr, reduce='mean')
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([1., 1., 1.]))
    """
    num_edges = edge_index[0].shape[0]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = np.empty(num_edges + 1, np.int64)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:]*=num_nodes
    idx[1:]+=(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        if isinstance(edge_index, Tensor):
            edge_index = edge_index[:, perm]
        elif isinstance(edge_index, tuple):
            edge_index = (edge_index[0][perm], edge_index[1][perm])
        else:
            raise NotImplementedError
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if edge_attr is None or isinstance(edge_attr, Tensor):
            return edge_index, edge_attr
        if isinstance(edge_attr, (list, tuple)):
            return edge_index, edge_attr
        return edge_index

    if isinstance(edge_index, Tensor):
        edge_index = edge_index[:, mask]
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][mask], edge_index[1][mask])
    else:
        raise NotImplementedError

    dim_size: Optional[int] = None
    if isinstance(edge_attr, (Tensor, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.size(1)
        idx = np.arange(0, num_edges)
        idx.sub_(mask.logical_not_().cumsum(dim=0))

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        if len(edge_attr) == 0:
            return edge_index, edge_attr
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index

import scipy

def is_torch_sparse_tensor(arr):
    return scipy.sparse.issparse(arr)


def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_attr = [[1, 2], [3, 4], [5, 6]]
        >>> edge_attr = torch.tensor(edge_attr)
        >>> remove_self_loops(edge_index, edge_attr)
        (tensor([[0, 1],
                [1, 0]]),
        tensor([[1, 2],
                [3, 4]]))
    """
    size: Optional[Tuple[int, int]] = None
    layout = None

    value: Optional[Tensor] = None
    if is_torch_sparse_tensor(edge_index):
        layout = edge_index.layout
        size = (edge_index.shape[0], edge_index.size(1))
        edge_index, value = to_edge_index(edge_index)

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if layout is not None:
        assert edge_attr is None
        assert value is not None
        value = value[mask]
        if str(layout) == 'torch.sparse_coo':  # str(...) for TorchScript :(
            return to_torch_coo_tensor(edge_index, value, size, True), None
        elif str(layout) == 'torch.sparse_csr':
            return to_torch_csr_tensor(edge_index, value, size, True), None
        raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]
