
import os.path as osp
import sys
from typing import Any, Dict, List, Literal, Optional, Union, overload
from uuid import uuid4

import fsspec
from graph_description.utils import get_dataset_folder

DEFAULT_CACHE_PATH = '/tmp/pyg_simplecache'


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


def normpath(path: str) -> str:
    if isdisk(path):
        return osp.normpath(path)
    return path


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


@overload
def ls(path: str, detail: Literal[False] = False) -> List[str]:
    pass


@overload
def ls(path: str, detail: Literal[True]) -> List[Dict[str, Any]]:
    pass


def ls(
    path: str,
    detail: bool = False,
) -> Union[List[str], List[Dict[str, Any]]]:
    fs = get_fs(path)
    outputs = fs.ls(path, detail=detail)

    if not isdisk(path):
        if detail:
            for output in outputs:
                output['name'] = fs.unstrip_protocol(output['name'])
        else:
            outputs = [fs.unstrip_protocol(output) for output in outputs]

    return outputs


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
            #home_dir = torch_geometric.get_home_dir()
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


def rm(path: str, recursive: bool = True) -> None:
    get_fs(path).rm(path, recursive)


def mv(path1: str, path2: str, recursive: bool = True) -> None:
    fs1 = get_fs(path1)
    fs2 = get_fs(path2)
    assert fs1.protocol == fs2.protocol
    fs1.mv(path1, path2, recursive)


def glob(path: str) -> List[str]:
    fs = get_fs(path)
    paths = fs.glob(path)

    if not isdisk(path):
        paths = [fs.unstrip_protocol(path) for path in paths]

    return paths

from scipy.sparse import coo_array
import numpy as np

def replace_sparse_matrices(d):
    d_out = {}
    for key, value in d.items():
        if isinstance(value, coo_array):
            d_out[key + "_sparse_row"] = value.row
            d_out[key + "_sparse_col"] = value.col
            d_out[key + "_sparse_data"] = value.data
            d_out[key + "_sparse_shape"] = np.array(value.shape, dtype=np.int64)
        else:
            d_out[key]=value
    return d_out

def undo_replace_sparse_matrices(d):
    d_out = {}
    for key, value in d.items():
        if key.endswith("_sparse_data"):
            prefix = key[:-len("_sparse_data")]
            row  = d[prefix + "_sparse_row"]
            col  = d[prefix + "_sparse_col"]
            data = d[prefix + "_sparse_data"]
            the_shape = d[prefix + "_sparse_shape"]
            d_out[prefix] = coo_array((data, (row, col)), shape=the_shape)
        elif   (key.endswith("_sparse_row") or
                key.endswith("_sparse_col") or
                key.endswith("_sparse_shape") ):
            continue
        else:
            d_out[key]=value
    return d_out


def replace_none(d):
    d_out = {}
    for key, value in d.items():
        if value is None or repr(value)=="None":
            d_out[key+"_None"] = 0
        else:
            d_out[key]=value
    return replace_sparse_matrices(d_out)

def undo_replace_none(d):
    d_out = {}
    for key, value in d.items():
        if key.endswith("_None"):
            d_out[key[:-5]] = None
        else:
            d_out[key]=value
    return undo_replace_sparse_matrices(d_out)

def torch_save(data: Any, path: str) -> None:
    import numpy as np
    #print("saving to", path)
    if data is None or data=="None":
        np.savez(path, None_data=data)
        return
    #print(data, type(data), path)
    d, slices = data
    #print(data, path)
    if slices is None:
        np.savez(path, **replace_none(d), slices_None=True)
    else:
        np.savez(path, **replace_none(d), slices=slices)
    return


def torch_load(path: str, map_location: Any = None) -> Any:
    import numpy as np
    load_result = np.load(path)
    if "None_data" in load_result:
        return "None"
    else:
        data = {key: value for key, value in load_result.items()}
        #print(data)
        if "slices_None" in data:
            slices = None
            del data["slices_None"]
        else:
            slices = load_result["slices"]
            del data["slices"]
        #print("loading", path)
        return undo_replace_none(data), slices
    #with fsspec.open(path, 'rb') as f:
    #    return np.load(f, map_location)