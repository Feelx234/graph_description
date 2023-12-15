from pathlib import Path
import pandas as pd
import pysubgroup as ps

import numpy as np

def get_dataset_folder():

    path = Path(__file__).parent.absolute()

    if str(path.name) == "scripts":
        path = path.parent
    if str(path.name) == "graph_description":
        path = path.parent
    if str(path.name) == "src":
        path = path.parent
    if str(path.name)!="datasets":
        path = path/"datasets"

    assert path.is_dir()
    return path


def prune_sparse_selectors(searchspace, df):
    out_sel = []
    for sel in searchspace:

        if not isinstance(sel, ps.EqualitySelector):
            out_sel.append(sel)
            continue
        if not isinstance(df[sel.attribute_name].dtype, pd.SparseDtype):
            out_sel.append(sel)
            continue
        if sel.attribute_value == df[sel.attribute_name].sparse.fill_value:
            continue
        #print(sel.attribute_value, df[sel.attribute_name].sparse.fill_value, df[sel.attribute_name].sparse.density)
        out_sel.append(sel)
    return out_sel


def random_split(labels, num_train_per_class=20, num_val=0.3, num_test=0.7):
    n_nodes = len(labels)
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    class_count = np.bincount(labels)
    min_class_count = class_count.min()
    if num_train_per_class > min_class_count:
        raise ValueError(
            f"Cannot choose num_train_per_class={num_train_per_class} as the smallest class has only {min_class_count} instances")
    num_classes = len(class_count)
    for c in range(num_classes):
        idx = (labels == c).nonzero()[0]
        idx = idx[np.random.permutation(idx.shape[0])[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero()[0]
    # pylint: disable-next=unsubscriptable-object, no-member
    remaining = remaining[np.random.permutation(remaining.shape[0])]

    make_round = False
    if isinstance(num_val, float) and isinstance(num_test, float) and num_val+num_test == 1.0:
        make_round=True
    if isinstance(num_val, float):
        assert num_val <= 1
        num_val = int( np.floor(num_val*len(remaining)))
    if isinstance(num_test, float):
        if make_round:
            num_test = len(remaining)-num_val
        else:
            num_test = int(np.floor(num_test*len(remaining)))
    val_mask[remaining[:num_val]] = True

    test_mask[remaining[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask