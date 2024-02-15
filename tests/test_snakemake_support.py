import unittest
import pandas as pd
import numpy as np
import networkx as nx
from graph_description.snakemake_support import _load_dataset_splitted
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal

class Test1(unittest.TestCase):
    def test_load_dataset_splitted(self):
        splits = {
            "train_mask" : np.array([1,0,1,1,0,0], dtype=bool),
            "val_mask" : np.array([0,1,0,0,1,0], dtype=bool),
            "test_mask" : np.array([0,0,0,0,0,1], dtype=bool),
        }
        np.random.seed(1)
        n = 6
        df = pd.DataFrame.from_dict({
            "x1" : np.random.rand(n),
            "x1" : np.random.rand(n),
            "x1" : np.random.rand(n),
            "labels": np.array([1,2,3,3,2,1], dtype=int)
        })
        labels_copy = df["labels"].copy()
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from([(0,1),(2,3), (5,4), (2,5), (3,5)])
        df_train, y_train, df_val, y_val, df_full = _load_dataset_splitted(splits, df,
                               labelsonly_dict=dict(output_path="some_prefix/split_labelsonly/other_stuff", G=G, round=1),
                                 return_full=True)

        assert_array_equal(df_full["labels"], labels_copy)
        assert_array_equal(df["labels"], labels_copy)
        assert_array_equal(df_full["neigh_sum(labels_-1)"], [1,0,1,1,1,1])
        assert_array_equal(df_full["neigh_sum(labels_1)"],  [0,1,0,0,0,0])
        assert_array_equal(df_full["neigh_sum(labels_3)"],  [0,0,1,1,0,2])

        assert_array_equal(df_full["neigh_mean(labels_-1)"], [1,0,0.5,0.5,1,1/3])
        assert_array_equal(df_full["neigh_mean(labels_1)"],  [0,1,0,0,0,0])
        assert_array_equal(df_full["neigh_mean(labels_3)"],  [0,0,0.5,0.5,0,2/3])

        assert_frame_equal(df_train, df_full[splits["train_mask"]].drop("labels", axis=1))
        assert_series_equal(y_train, df_full[splits["train_mask"]]["labels"])

        assert_frame_equal(df_val, df_full[splits["val_mask"]].drop("labels", axis=1))
        assert_series_equal(y_val, df_full[splits["val_mask"]]["labels"])

        edges = np.array(list(G.edges))
        df_train, y_train, df_val, y_val, df_full = _load_dataset_splitted(splits, df,
                               labelsonly_dict=dict(output_path="some_prefix/split_labelsonly/other_stuff", G=edges, round=1),
                                 return_full=True)


if __name__ == "__main__":
    unittest.main()