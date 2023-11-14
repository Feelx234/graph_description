from pathlib import Path
import pandas as pd
import pysubgroup as ps

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