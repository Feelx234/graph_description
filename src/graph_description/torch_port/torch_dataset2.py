import copy
import os.path as osp
import warnings
from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import numpy as np
#import torch
Tensor = np.array
from tqdm import tqdm

#from torch_geometric.data import Batch, Data
#from torch_geometric.data.collate import collate
#from torch_geometric.data.data import BaseData
#from torch_geometric.data.dataset import Dataset, IndexType
#from torch_geometric.data.separate import separate
#from torch_geometric.io import fs
from graph_description.torch_port.torch_utils import Data

import copy
import os.path as osp
import re
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

IndexType = Union[slice, np.ndarray, Sequence]
MISSING = '???'
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    overload,
)
from typing_extensions import Self
import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain

from graph_description.torch_port import fs

def is_floating_point(x):
    if isinstance(x, float):
        return True
    if isinstance(x, np.floating):
        return True


class BaseData:
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def stores_as(self, data: Self):
        raise NotImplementedError

    @property
    def stores(self) -> List[Any]:
        raise NotImplementedError

    @property
    def node_stores(self) -> List[Any]:
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[Any]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        raise NotImplementedError

    def update(self, data: Self) -> Self:
        r"""Updates the data object with the elements from another data object.
        Added elements will override existing ones (in case of duplicates).
        """
        raise NotImplementedError

    def concat(self, data: Self) -> Self:
        r"""Concatenates :obj:`self` with another :obj:`data` object.
        All values needs to have matching shapes at non-concat dimensions.
        """
        out = copy.copy(self)
        for store, other_store in zip(out.stores, data.stores):
            store.concat(other_store)
        return out

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the dimension for which the value :obj:`value` of the
        attribute :obj:`key` will get concatenated when creating mini-batches
        using :class:`torch_geometric.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        :obj:`value` of the attribute :obj:`key` when creating mini-batches
        using :class:`torch_geometric.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def debug(self):
        raise NotImplementedError

    ###########################################################################

    def keys(self) -> List[str]:
        r"""Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data.
        """
        return key in self.keys()

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph.

        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            :pyg:`PyG` then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            return sum([v.num_nodes for v in self.node_stores])
        except TypeError:
            return None

    @overload
    def size(self) -> Tuple[Optional[int], Optional[int]]:
        pass

    @overload
    def size(self, dim: int) -> Optional[int]:
        pass

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""Returns the size of the adjacency matrix induced by the graph."""
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    @property
    def num_edges(self) -> int:
        r"""Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        return sum([v.num_edges for v in self.edge_stores])

    def node_attrs(self) -> List[str]:
        r"""Returns all node-level tensor attribute names."""
        return list(set(chain(*[s.node_attrs() for s in self.node_stores])))

    def edge_attrs(self) -> List[str]:
        r"""Returns all edge-level tensor attribute names."""
        return list(set(chain(*[s.edge_attrs() for s in self.edge_stores])))

    @property
    def node_offsets(self) -> Dict[Any, int]:
        out: Dict[Any, int] = {}
        offset: int = 0
        for store in self.node_stores:
            out[store._key] = offset
            offset = offset + store.num_nodes
        return out

    def generate_ids(self):
        r"""Generates and sets :obj:`n_id` and :obj:`e_id` attributes to assign
        each node and edge to a continuously ascending and unique ID.
        """
        for store in self.node_stores:
            store.n_id = np.arange(store.num_nodes)
        for store in self.edge_stores:
            store.e_id = np.arange(store.num_edges)

    def is_sorted(self, sort_by_row: bool = True) -> bool:
        r"""Returns :obj:`True` if edge indices :obj:`edge_index` are sorted.

        Args:
            sort_by_row (bool, optional): If set to :obj:`False`, will require
                column-wise order/by destination node order of
                :obj:`edge_index`. (default: :obj:`True`)
        """
        return all(
            [store.is_sorted(sort_by_row) for store in self.edge_stores])

    def sort(self, sort_by_row: bool = True) -> Self:
        r"""Sorts edge indices :obj:`edge_index` and their corresponding edge
        features.

        Args:
            sort_by_row (bool, optional): If set to :obj:`False`, will sort
                :obj:`edge_index` in column-wise order/by destination node.
                (default: :obj:`True`)
        """
        out = copy.copy(self)
        for store in out.edge_stores:
            store.sort(sort_by_row)
        return out

    def is_coalesced(self) -> bool:
        r"""Returns :obj:`True` if edge indices :obj:`edge_index` are sorted
        and do not contain duplicate entries.
        """
        return all([store.is_coalesced() for store in self.edge_stores])

    def coalesce(self) -> Self:
        r"""Sorts and removes duplicated entries from edge indices
        :obj:`edge_index`.
        """
        out = copy.copy(self)
        for store in out.edge_stores:
            store.coalesce()
        return out

    def is_sorted_by_time(self) -> bool:
        r"""Returns :obj:`True` if :obj:`time` is sorted."""
        return all([store.is_sorted_by_time() for store in self.stores])

    def sort_by_time(self) -> Self:
        r"""Sorts data associated with :obj:`time` according to :obj:`time`."""
        out = copy.copy(self)
        for store in out.stores:
            store.sort_by_time()
        return out

    def snapshot(
        self,
        start_time: Union[float, int],
        end_time: Union[float, int],
    ) -> Self:
        r"""Returns a snapshot of :obj:`data` to only hold events that occurred
        in period :obj:`[start_time, end_time]`.
        """
        out = copy.copy(self)
        for store in out.stores:
            store.snapshot(start_time, end_time)
        return out

    def up_to(self, end_time: Union[float, int]) -> Self:
        r"""Returns a snapshot of :obj:`data` to only hold events that occurred
        up to :obj:`end_time` (inclusive of :obj:`edge_time`).
        """
        out = copy.copy(self)
        for store in out.stores:
            store.up_to(end_time)
        return out

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True` if the graph contains isolated nodes."""
        return any([store.has_isolated_nodes() for store in self.edge_stores])

    def has_self_loops(self) -> bool:
        """Returns :obj:`True` if the graph contains self-loops."""
        return any([store.has_self_loops() for store in self.edge_stores])

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True` if graph edges are undirected."""
        return all([store.is_undirected() for store in self.edge_stores])

    def is_directed(self) -> bool:
        r"""Returns :obj:`True` if graph edges are directed."""
        return not self.is_undirected()

    def apply_(self, func: Callable, *args: str):
        r"""Applies the in-place function :obj:`func`, either to all attributes
        or only the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply_(func, *args)
        return self

    def apply(self, func: Callable, *args: str):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply(func, *args)
        return self

    def clone(self, *args: str):
        r"""Performs cloning of tensors, either for all attributes or only the
        ones given in :obj:`*args`.
        """
        return copy.copy(self).apply(lambda x: x.clone(), *args)

    def contiguous(self, *args: str):
        r"""Ensures a contiguous memory layout, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.contiguous(), *args)

    def to(self, device: Union[int, str], *args: str,
           non_blocking: bool = False):
        r"""Performs tensor device conversion, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: str):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(self, device: Optional[Union[int, str]] = None, *args: str,
             non_blocking: bool = False):
        r"""Copies attributes to CUDA memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        # Some PyTorch tensor like objects require a default value for `cuda`:
        device = 'cuda' if device is None else device
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking),
                          *args)

    def pin_memory(self, *args: str):
        r"""Copies attributes to pinned memory, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: str):
        r"""Moves attributes to shared memory, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply_(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: str):
        r"""Detaches attributes from the computation graph, either for all
        attributes or only the ones given in :obj:`*args`.
        """
        return self.apply_(lambda x: x.detach_(), *args)

    def detach(self, *args: str):
        r"""Detaches attributes from the computation graph by creating a new
        tensor, either for all attributes or only the ones given in
        :obj:`*args`.
        """
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: str, requires_grad: bool = True):
        r"""Tracks gradient computation, either for all attributes or only the
        ones given in :obj:`*args`.
        """
        return self.apply_(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)

    def record_stream(self, stream, *args: str):
        r"""Ensures that the tensor memory is not reused for another tensor
        until all current work queued on :obj:`stream` has been completed,
        either for all attributes or only the ones given in :obj:`*args`.
        """
        return self.apply_(lambda x: x.record_stream(stream), *args)

    @property
    def is_cuda(self) -> bool:
        r"""Returns :obj:`True` if any :class:`torch.Tensor` attribute is
        stored on the GPU, :obj:`False` otherwise.
        """
        for store in self.stores:
            for value in store.values():
                if isinstance(value, Tensor) and value.is_cuda:
                    return True
        return False



###############################################################################



class Dataset(ABC):
    r"""Dataset base class for creating graph datasets.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_dataset.html>`__ for the accompanying tutorial.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    @abstractmethod
    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def get(self, idx: int) -> BaseData:
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = osp.expanduser(fs.normpath(root))

        self.root = root or MISSING
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self._indices: Optional[Sequence] = None
        self.force_reload = force_reload

        if self.has_download:
            self._download()

        if self.has_process:
            self._process()

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list[0] = None
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the dataset.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list[0] = None
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")

    def _infer_num_classes(self, y) -> int:
        if y is None:
            return 0
        elif not is_floating_point(y):
            return int(y.max()) + 1
        elif is_floating_point(y):
            num_classes = np.unique(y).numel()
            if num_classes > 2:
                warnings.warn("Found floating-point labels while calling "
                              "`dataset.num_classes`. Returning the number of "
                              "unique elements. Please make sure that this "
                              "is expected before proceeding.")
            return num_classes
        else:
            return y.size(-1)

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        # We iterate over the dataset and collect all labels to determine the
        # maximum number of classes. Importantly, in rare cases, `__getitem__`
        # may produce a tuple of data objects (e.g., when used in combination
        # with `RandomLinkSplit`, so we take care of this case here as well:
        print("AAA")
        data_list = _get_flattened_data_list([data for data in self])
        if 'y' in data_list[0] and isinstance(data_list[0].y, np.array):
            y = np.cat([data.y for data in data_list if 'y' in data], dim=0)
        else:
            y = np.as_array([data.y for data in data_list if 'y' in data])
        print("AAAAAA")
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(y)

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing.
        """
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]

    @property
    def has_download(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`download` method."""
        return overrides_method(self.__class__, 'download')

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    @property
    def has_process(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`process` method."""
        return overrides_method(self.__class__, 'process')

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.npz')
        if osp.exists(f) and fs.torch_load(f) != _repr(self.pre_transform):
            warnings.warn(
                "The `pre_transform` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-processing technique, pass "
                "`force_reload=True` explicitly to reload the dataset.")

        f = osp.join(self.processed_dir, 'pre_filter.npz')
        if osp.exists(f) and fs.torch_load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, pass "
                "`force_reload=True` explicitly to reload the dataset.")

        if not self.force_reload and files_exist(self.processed_paths):
            return

        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.npz')
        fs.torch_save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.npz')
        fs.torch_save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', BaseData]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, np.array) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def index_select(self, idx: IndexType) -> 'Dataset':
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union['Dataset', Tuple['Dataset', Tensor]]:
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will also
                return the random permutation used to shuffle the dataset.
                (default: :obj:`False`)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr})'

    def get_summary(self):
        r"""Collects summary statistics for the dataset."""
        from torch_geometric.data.summary import Summary
        return Summary.from_dataset(self)

    def print_summary(self):  # pragma: no cover
        r"""Prints summary statistics of the dataset to the console."""
        print(str(self.get_summary()))

    def to_datapipe(self):
        r"""Converts the dataset into a :class:`torch.utils.data.DataPipe`.

        The returned instance can then be used with :pyg:`PyG's` built-in
        :class:`DataPipes` for baching graphs as follows:

        .. code-block:: python

            from torch_geometric.datasets import QM9

            dp = QM9(root='./data/QM9/').to_datapipe()
            dp = dp.batch_graphs(batch_size=2, drop_last=True)

            for batch in dp:
                pass

        See the `PyTorch tutorial
        <https://pytorch.org/data/main/tutorial.html>`_ for further background
        on DataPipes.
        """
        from torch_geometric.data.datapipes import DatasetAdapter

        return DatasetAdapter(self)


def overrides_method(cls, method_name: str):

    if method_name in cls.__dict__:
        return True

    out = False
    for base in cls.__bases__:
        if base != Dataset and base != InMemoryDataset:
            out |= overrides_method(base, method_name)
    return out


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([fs.exists(f) for f in files])


def _repr(obj: Any) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', str(obj))


def _get_flattened_data_list(data_list: List[Any]) -> List[BaseData]:
    outs: List[BaseData] = []
    for data in data_list:
        if isinstance(data, BaseData):
            outs.append(data)
        elif isinstance(data, (tuple, list)):
            outs.extend(_get_flattened_data_list(data))
        elif isinstance(data, dict):
            outs.extend(_get_flattened_data_list(data.values()))
    return outs

class InMemoryDataset(Dataset, ABC):
    r"""Dataset base class for creating graph datasets which easily fit
    into CPU memory.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log,
                         force_reload)

        self._data: Optional[BaseData] = None
        self.slices: Optional[Dict[str, Tensor]] = None
        self._data_list: Optional[MutableSequence[Optional[BaseData]]] = None

    @property
    def num_classes(self) -> int:
        if self.transform is None:
            return self._infer_num_classes(self._data.y)
        return super().num_classes

    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get(self, idx: int) -> BaseData:
        # TODO (matthias) Avoid unnecessary copy here.
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data

    @classmethod
    def save(cls, data_list: Sequence[BaseData], path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        #if self.log:
        #    print("saving to "+path)
        fs.torch_save((data._asdict(), slices), path)

    def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        data, self.slices = fs.torch_load(path)
        if isinstance(data, dict):  # Backward compatibility.
            #data = data_cls.from_dict(data)
            data = data_cls(**data)
        self.data = data

    @staticmethod
    def collate(
        data_list: Sequence[BaseData],
    ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:
        r"""Collates a list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects to the internal
        storage format of :class:`~torch_geometric.data.InMemoryDataset`.
        """
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if idx is None:
            data_list = [self.get(i) for i in self.indices()]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = None
        dataset.data, dataset.slices = self.collate(data_list)
        return dataset

    def to_on_disk_dataset(
        self,
        root: Optional[str] = None,
        backend: str = 'sqlite',
        log: bool = True,
    ) -> 'torch_geometric.data.OnDiskDataset':
        r"""Converts the :class:`InMemoryDataset` to a :class:`OnDiskDataset`
        variant. Useful for distributed training and hardware instances with
        limited amount of shared memory.

        root (str, optional): Root directory where the dataset should be saved.
            If set to :obj:`None`, will save the dataset in
            :obj:`root/on_disk`.
            Note that it is important to specify :obj:`root` to account for
            different dataset splits. (optional: :obj:`None`)
        backend (str): The :class:`Database` backend to use.
            (default: :obj:`"sqlite"`)
        log (bool, optional): Whether to print any console output while
            processing the dataset. (default: :obj:`True`)
        """
        if root is None and (self.root is None or not osp.exists(self.root)):
            raise ValueError(f"The root directory of "
                             f"'{self.__class__.__name__}' is not specified. "
                             f"Please pass in 'root' when creating on-disk "
                             f"datasets from it.")

        root = root or osp.join(self.root, 'on_disk')

        in_memory_dataset = self
        ref_data = in_memory_dataset.get(0)
        if not isinstance(ref_data, Data):
            raise NotImplementedError(
                f"`{self.__class__.__name__}.to_on_disk_dataset()` is "
                f"currently only supported on homogeneous graphs")

        # Parse the schema ====================================================

        schema: Dict[str, Any] = {}
        for key, value in ref_data.to_dict().items():
            if isinstance(value, (int, float, str)):
                schema[key] = value.__class__
            elif isinstance(value, Tensor) and value.dim() == 0:
                schema[key] = dict(dtype=value.dtype, size=(-1, ))
            elif isinstance(value, Tensor):
                size = list(value.size())
                size[ref_data.__cat_dim__(key, value)] = -1
                schema[key] = dict(dtype=value.dtype, size=tuple(size))
            else:
                schema[key] = object

        # Create the on-disk dataset ==========================================

        class OnDiskDataset(torch_geometric.data.OnDiskDataset):
            def __init__(
                self,
                root: str,
                transform: Optional[Callable] = None,
            ):
                super().__init__(
                    root=root,
                    transform=transform,
                    backend=backend,
                    schema=schema,
                )

            def process(self):
                _iter = [
                    in_memory_dataset.get(i)
                    for i in in_memory_dataset.indices()
                ]
                if log:  # pragma: no cover
                    _iter = tqdm(_iter, desc='Converting to OnDiskDataset')

                data_list: List[Data] = []
                for i, data in enumerate(_iter):
                    data_list.append(data)
                    if i + 1 == len(in_memory_dataset) or (i + 1) % 1000 == 0:
                        self.extend(data_list)
                        data_list = []

            def serialize(self, data: Data) -> Dict[str, Any]:
                return data.to_dict()

            def deserialize(self, data: Dict[str, Any]) -> Data:
                return Data.from_dict(data)

            def __repr__(self) -> str:
                arg_repr = str(len(self)) if len(self) > 1 else ''
                return (f'OnDisk{in_memory_dataset.__class__.__name__}('
                        f'{arg_repr})')

        return OnDiskDataset(root, transform=in_memory_dataset.transform)

    @property
    def data(self) -> Any:
        msg1 = ("It is not recommended to directly access the internal "
                "storage format `data` of an 'InMemoryDataset'.")
        msg2 = ("The given 'InMemoryDataset' only references a subset of "
                "examples of the full dataset, but 'data' will contain "
                "information of the full dataset.")
        msg3 = ("The data of the dataset is already cached, so any "
                "modifications to `data` will not be reflected when accessing "
                "its elements. Clearing the cache now by removing all "
                "elements in `dataset._data_list`.")
        msg4 = ("If you are absolutely certain what you are doing, access the "
                "internal storage via `InMemoryDataset._data` instead to "
                "suppress this warning. Alternatively, you can access stacked "
                "individual attributes of every graph via "
                "`dataset.{attr_name}`.")

        msg = msg1
        if self._indices is not None:
            msg += f' {msg2}'
        if self._data_list is not None:
            msg += f' {msg3}'
            self._data_list = None
        msg += f' {msg4}'

        warnings.warn(msg)

        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._data_list = None

    #def __getattr__(self, key: str) -> Any:
    #    data = self.__dict__.get('_data')
    #    if isinstance(data, Data) and key in data._fields:
    #        if self._indices is None and data.__inc__(key, data[key]) == 0:
    #            return data[key]
    #        else:
    #            data_list = [self.get(i) for i in self.indices()]
    #            return Batch.from_data_list(data_list)[key]#

    #    raise AttributeError(f"'{self.__class__.__name__}' object has no "
    #                         f"attribute '{key}'")

    def to(self, device: Union[int, str]) -> 'InMemoryDataset':
        r"""Performs device conversion of the whole dataset."""
        if self._indices is not None:
            raise ValueError("The given 'InMemoryDataset' only references a "
                             "subset of examples of the full dataset")
        if self._data_list is not None:
            raise ValueError("The data of the dataset is already cached")
        self._data.to(device)
        return self

    def cpu(self, *args: str) -> 'InMemoryDataset':
        r"""Moves the dataset to CPU memory."""
        return self.to(torch.device('cpu'))

    def cuda(
        self,
        device: Optional[Union[int, str]] = None,
    ) -> 'InMemoryDataset':
        r"""Moves the dataset toto CUDA memory."""
        if isinstance(device, int):
            device = f'cuda:{int}'
        elif device is None:
            device = 'cuda'
        return self.to(device)


def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):
        for key, value in node.items():
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node