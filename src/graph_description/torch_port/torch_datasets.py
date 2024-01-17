import os.path as osp
from typing import Callable, List, Optional

from graph_description.utils import get_dataset_folder
import numpy as np
import sys
import fsspec













import os.path as osp
import warnings
from itertools import repeat
from typing import Dict, List, Optional, Any
from collections import namedtuple

from scipy.sparse import coo_array
#from torch_geometric.typing import SparseTensor
#from torch_geometric.utils import coalesce, index_to_mask, remove_self_loops

from graph_description.torch_port.torch_utils import index_to_mask, coalesce, remove_self_loops, Data

try:
    import cPickle as pickle
except ImportError:
    import pickle


from typing import List, Optional
from graph_description.torch_port import fs

from graph_description.torch_port.torch_dataset2 import InMemoryDataset

def parse_txt_array(
    src: List[str],
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype = None,
    device = None,
):
    empty = np.empty(0, dtype=dtype)
    to_number = float if isinstance(empty, np.floating) else int

    return np.array([[to_number(x) for x in line.split(sep)[start:end]]
                         for line in src], dtype=dtype)


def read_txt_array(
    path: str,
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype = None,
    device = None,
):
    with fsspec.open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def read_planetoid_data(folder: str, prefix: str, split:str):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]

    x, tx, allx, y, ty, ally, graph, test_index = items
    test_index = test_index.ravel()
    train_index = np.arange(y.shape[0], dtype=np.int64)
    val_index = np.arange(y.shape[0], y.shape[0] + 500, dtype=np.int64)
    sorted_test_index = test_index.copy()
    sorted_test_index.sort()
    sorted_test_index = sorted_test_index.ravel()#[0]

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = int(test_index.max() - test_index.min()) + 1

        tx_ext = np.zeros((len_test_indices, tx.shape[1]), dtype=tx.dtype)
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = np.zeros((len_test_indices, ty.shape[1]), dtype=ty.dtype)
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == 'nell.0.001':
        tx_ext = np.zeros(len(graph) - allx.shape[0], x.shape[1])
        tx_ext[sorted_test_index - allx.shape[0]] = tx

        ty_ext = np.zeros((len(graph) - ally.shape[0], y.shape[1]), dtype=np.int64)
        ty_ext[sorted_test_index - ally.shape[0]] = ty

        tx, ty = tx_ext, ty_ext

        x = np.concatenate([allx, tx], axis=0)
        x[test_index] = x[sorted_test_index]

        # Creating feature vectors for relations.
        #SparseTensor.from_dense(x).coo()
        sparse_mat = coo_array(x)
        row, col, value = sparse_mat.row, sparse_mat.col, sparse_mat.data
        rows, cols, values = [row], [col], [value]

        mask1 = index_to_mask(test_index, size=len(graph))
        mask2 = index_to_mask(np.arange(allx.shape[0], len(graph)),
                              size=len(graph))
        mask = ~mask1 | ~mask2
        isolated_index = mask.nonzero(as_tuple=False).view(-1)[allx.shape[0]:]

        rows += [isolated_index]
        cols += [np.arange(isolated_index.shape[0]) + x.shape[1]]
        values += [np.ones(isolated_index.shape[0])]

        #x = SparseTensor(row=np.concatenate(rows), col=np.concatenate(cols),
        #                 value=np.concatenate(values))
        x = coo_array((np.concatenate(values), (np.concatenate(cols), np.concatenate(rows))), shape=x.shape)
    else:
        x = np.concatenate([allx, tx], axis=0)
        x[test_index] = x[sorted_test_index]

    y = np.argmax(np.concatenate([ally, ty], axis=0),axis=1)#.max(axis=1)#[1]
    y[test_index] = y[sorted_test_index]
    y = np.array(y, dtype=np.int64)

    if split.lower() == "geom-gcn":
        def my_repeat(arr):
            return np.repeat(arr.reshape(y.shape[0],1), 10, axis=1)
        train_mask = my_repeat(index_to_mask(train_index, size=y.shape[0]))
        val_mask = my_repeat(index_to_mask(val_index, size=y.shape[0]))
        test_mask = my_repeat(index_to_mask(test_index, size=y.shape[0]))
    else:
        train_mask = index_to_mask(train_index, size=y.shape[0])
        val_mask = index_to_mask(val_index, size=y.shape[0])
        test_mask = index_to_mask(test_index, size=y.shape[0])


    edge_index = edge_index_from_dict(
        graph_dict=graph,  # type: ignore
        num_nodes=y.shape[0],
    )

    data = Data(x, y, edge_index, train_mask, val_mask, test_mask)

    return data


def read_file(folder: str, prefix: str, name: str):
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path, dtype=np.int64)

    with fsspec.open(path, 'rb') as f:
        warnings.filterwarnings('ignore', '.*`scipy.sparse.csr` name.*')
        out = pickle.load(f, encoding='latin1')

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = np.array(out).astype(np.float64)
    return out


def edge_index_from_dict(
    graph_dict: Dict[int, List[int]],
    num_nodes: Optional[int] = None,
):
    rows: List[int] = []
    cols: List[int] = []
    for key, value in graph_dict.items():
        rows += repeat(key, len(value))
        cols += value
    edge_index = np.stack([np.array(rows), np.array(cols)], axis=0)

    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index



def get_fs(path: str) -> fsspec.AbstractFileSystem:
    r"""Get filesystem backend given a path URI to the resource.

    Here are some common example paths and dispatch result:

    * :obj:`"/home/file"` ->
      :class:`fsspec.implementations.local.LocalFileSystem`
    * :obj:`"memory://home/file"` ->
      :class:`fsspec.implementations.memory.MemoryFileSystem`
    * :obj:`"https://home/file"` ->
      :class:`fsspec.implementations.http.HTTPFileSystem`
    * :obj:`"gs://home/file"` -> :class:`gcsfs.GCSFileSystem`
    * :obj:`"s3://home/file"` -> :class:`s3fs.S3FileSystem`

    A full list of supported backend implementations of :class:`fsspec` can be
    found `here <https://github.com/fsspec/filesystem_spec/blob/master/fsspec/
    registry.py#L62>`_.

    The backend dispatch logic can be updated with custom backends following
    `this tutorial <https://filesystem-spec.readthedocs.io/en/latest/
    developer.html#implementing-a-backend>`_.

    Args:
        path (str): The URI to the filesystem location, *e.g.*,
            :obj:`"gs://home/me/file"`, :obj:`"s3://..."`.
    """
    return fsspec.core.url_to_fs(path)[0]

def exists(path: str) -> bool:
    return get_fs(path).exists(path)


def makedirs(path: str, exist_ok: bool = True) -> None:
    return get_fs(path).makedirs(path, exist_ok)


def isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def isfile(path: str) -> bool:
    return get_fs(path).isfile(path)


def isdisk(path: str) -> bool:
    return 'file' in get_fs(path).protocol


def islocal(path: str) -> bool:
    return isdisk(path) or 'memory' in get_fs(path).protocol

def cp(
    path1: str,
    path2: str,
    extract: bool = False,
    log: bool = True,
    use_cache: bool = True,
    clear_cache: bool = True,
) -> None:
    kwargs: Dict[str, Any] = {}

    is_path1_dir = isdir(path1)
    is_path2_dir = isdir(path2)

    # Cache result if the protocol is not local:
    cache_dir: Optional[str] = None
    if not islocal(path1):
        if log and 'pytest' not in sys.modules:
            print(f'Downloading {path1}', file=sys.stderr)

        if extract and use_cache:  # Cache seems to confuse the gcs filesystem.
            home_dir = get_dataset_folder()
            cache_dir = osp.join(home_dir, 'simplecache', uuid4().hex)
            kwargs.setdefault('simplecache', dict(cache_storage=cache_dir))
            path1 = f'simplecache::{path1}'

    # Handle automatic extraction:
    multiple_files = False
    if extract and path1.endswith('.tar.gz'):
        kwargs.setdefault('tar', dict(compression='gzip'))
        path1 = f'tar://**::{path1}'
        multiple_files = True
    elif extract and path1.endswith('.zip'):
        path1 = f'zip://**::{path1}'
        multiple_files = True
    elif extract and path1.endswith('.gz'):
        kwargs.setdefault('compression', 'infer')
    elif extract:
        raise NotImplementedError(
            f"Automatic extraction of '{path1}' not yet supported")

    # If the source path points to a directory, we need to make sure to
    # recursively copy all files within this directory. Additionally, if the
    # destination folder does not yet exist, we inherit the basename from the
    # source folder.
    if is_path1_dir:
        if exists(path2):
            path2 = osp.join(path2, osp.basename(path1))
        path1 = osp.join(path1, '**')
        multiple_files = True

    # Perform the copy:
    for open_file in fsspec.open_files(path1, **kwargs):
        with open_file as f_from:
            if not multiple_files:
                if is_path2_dir:
                    basename = osp.basename(path1)
                    if extract and path1.endswith('.gz'):
                        basename = '.'.join(basename.split('.')[:-1])
                    to_path = osp.join(path2, basename)
                else:
                    to_path = path2
            else:
                # Open file has protocol stripped.
                common_path = osp.commonprefix(
                    [fsspec.core.strip_protocol(path1), open_file.path])
                to_path = osp.join(path2, open_file.path[len(common_path):])
            with fsspec.open(to_path, 'wb') as f_to:
                while True:
                    chunk = f_from.read(10 * 1024 * 1024)
                    if not chunk:
                        break
                    f_to.write(chunk)

    if use_cache and clear_cache and cache_dir is not None:
        try:
            rm(cache_dir)
        except PermissionError:  # FIXME
            # Windows test yield "PermissionError: The process cannot access
            # the file because it is being used by another process"
            # This is a quick workaround until we figure out the deeper issue.
            pass


class Planetoid(InMemoryDataset):
    r"""The citation network datasets :obj:`"Cora"`, :obj:`"CiteSeer"` and
    :obj:`"PubMed"` from the `"Revisiting Semi-Supervised Learning with Graph
    Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`,
            :obj:`"PubMed"`).
        split (str, optional): The type of dataset split (:obj:`"public"`,
            :obj:`"full"`, :obj:`"geom-gcn"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
            `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Cora
          - 2,708
          - 10,556
          - 1,433
          - 7
        * - CiteSeer
          - 3,327
          - 9,104
          - 3,703
          - 6
        * - PubMed
          - 19,717
          - 88,648
          - 500
          - 3
    """
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        log : bool = True
    ) -> None:
        self.name = name

        self.split = split.lower()
        assert self.split in ['public', 'full', 'geom-gcn', 'random']

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload, log=log)
        self.load(self.processed_paths[0])

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask[:] = False
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero()[0]
                idx = idx[np.random.permutation(idx.shape[0])[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero()[0]
            remaining = remaining[np.random.permutation(remaining.shape[0])]

            data.val_mask[:]=(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask[:]=(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'processed')
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.npz'

    def download(self) -> None:
        for name in self.raw_file_names:
            fs.cp(f'{self.url}/{name}', self.raw_dir)
        if self.split == 'geom-gcn':
            for i in range(10):
                url = f'{self.geom_gcn_url}/splits/{self.name.lower()}'
                fs.cp(f'{url}_split_0.6_0.2_{i}.npz', self.raw_dir)

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, self.name, self.split)

        if self.split == 'geom-gcn':
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(splits['train_mask'])
                val_masks.append(splits['val_mask'])
                test_masks.append(splits['test_mask'])
            data.train_mask[:] = np.stack(train_masks, axis=1)
            data.val_mask[:] = np.stack(val_masks, axis=1)
            data.test_mask[:] = np.stack(test_masks, axis=1)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


import scipy.sparse as sp

def read_npz(path: str, to_undirected: bool = True) -> Data:
    with np.load(path) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Data:
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = np.array(x, dtype=np.float64)
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    row = np.array(adj.row, dtype=np.int64)
    col = np.array(adj.col, dtype=np.int64)
    edge_index = np.stack([row, col], axis=0)
    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.shape[0])

    y = np.array(f['labels'], dtype=np.int64)

    return Data(x=x, edge_index=edge_index, y=y)

from typing import List, Optional, Tuple, Union

Tensor = type(np.array(0))
def to_undirected_fn(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor], str] = '???',
    num_nodes: Optional[int] = None,
    reduce: str = 'add',
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> to_undirected(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> edge_weight = torch.tensor([1., 1., 1.])
        >>> to_undirected(edge_index, edge_weight)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([2., 2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>>  to_undirected(edge_index, edge_weight, reduce='mean')
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([1., 1., 1., 1.]))
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        num_nodes = edge_attr
        edge_attr = '???'

    row, col = edge_index[0], edge_index[1]
    row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
    edge_index = np.stack([row, col], axis=0)

    if isinstance(edge_attr, Tensor):
        edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [np.concatenate([e, e], axis=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)

import os
import os.path as osp
import ssl
import urllib
import urllib.request

def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if fs.exists(path):  # pragma: no cover
        if log and 'pytest' not in sys.modules:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path



class CitationFull(InMemoryDataset):
    r"""The full citation network datasets from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.
    Datasets include :obj:`"Cora"`, :obj:`"Cora_ML"`, :obj:`"CiteSeer"`,
    :obj:`"DBLP"`, :obj:`"PubMed"`.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
            :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        to_undirected (bool, optional): Whether the original graph is
            converted to an undirected one. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Cora
          - 19,793
          - 126,842
          - 8,710
          - 70
        * - Cora_ML
          - 2,995
          - 16,316
          - 2,879
          - 7
        * - CiteSeer
          - 4,230
          - 10,674
          - 602
          - 6
        * - DBLP
          - 17,716
          - 105,734
          - 1,639
          - 4
        * - PubMed
          - 19,717
          - 88,648
          - 500
          - 3
    """

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        to_undirected: bool = True,
        force_reload: bool = False,
        log : bool = True,
        split = "public"
    ) -> None:
        self.name = name.lower()
        self.to_undirected = to_undirected
        assert split is None or split =="public"
        assert self.name in ['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload, log=log)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        suffix = 'undirected' if self.to_undirected else 'directed'
        return f'data_{suffix}.npz'

    def download(self) -> None:
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self) -> None:
        data = read_npz(self.raw_paths[0], to_undirected=self.to_undirected)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'



import json
from itertools import chain

class WikiCS(InMemoryDataset):
    r"""The semi-supervised Wikipedia-based dataset from the
    `"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks"
    <https://arxiv.org/abs/2007.02901>`_ paper, containing 11,701 nodes,
    216,123 edges, 10 classes and 20 different training splits.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        is_undirected (bool, optional): Whether the graph is undirected.
            (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(
        self,
        root: str,
        name: str="wikics",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        is_undirected: Optional[bool] = None,
        force_reload: bool = False,
                log : bool = True,
                split = "public"
    ) -> None:
        if is_undirected is None:
            #warnings.warn(
            #    f"The {self.__class__.__name__} dataset now returns an "
            #    f"undirected graph by default. Please explicitly specify "
            #    f"'is_undirected=False' to restore the old behavior.")
            is_undirected = True
        self.is_undirected = is_undirected
        self.name="wikics"
        assert split == "public"
        assert name.lower()=="wikics"
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload, log=log)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.json']

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data_undirected.npz' if self.is_undirected else 'data.npz'

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], 'r') as f:
            data = json.load(f)

        x = np.array(data['features'], dtype=np.float64)
        y = np.array(data['labels'], dtype=np.int64)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))  # type: ignore
        edge_index = np.array(edges, dtype=np.int64)
        if self.is_undirected:
            edge_index = to_undirected_fn(edge_index, num_nodes=x.shape[0])

        train_mask = np.array(data['train_masks'], dtype=bool)
        #train_mask = train_mask.t().contiguous()

        val_mask = np.array(data['val_masks'], dtype=bool)
        #val_mask = val_mask.t().contiguous()

        test_mask = np.array(data['test_mask'], dtype=bool)

        stopping_mask = np.array(data['stopping_masks'], dtype=bool)
        #stopping_mask = stopping_mask.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    stopping_mask=stopping_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])