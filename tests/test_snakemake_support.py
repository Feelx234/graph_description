import unittest
import pandas as pd
import numpy as np
import networkx as nx
from graph_description.snakemake_support import _load_dataset_splitted


class Test1(unittest.TestCase):
    def test_load_dataset_splitted(self):
        splits = {
            "train_mask" : np.array([1,0,0,1,0,0], dtype=bool),
            "val_mask" : np.array([0,1,0,0,1,0], dtype=bool),
            "test_mask" : np.array([0,0,1,0,0,1], dtype=bool),
        }
        np.random.seed(1)
        n = 6
        df = pd.DataFrame.from_dict({
            "x1" : np.random.rand(n),
            "x1" : np.random.rand(n),
            "x1" : np.random.rand(n),
            "labels": np.array([1,2,3,1,2,3], dtype=int)
        })
        nx.DiGraph()
        _load_dataset_splitted(splits, df, labelsonly_dict=dict(output_path="split_labelsonly", G=))