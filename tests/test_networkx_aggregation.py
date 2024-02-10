import unittest
from functools import partial
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from graph_description.networkx_aggregation import SumAggregator, MeanAggregator, apply_aggregator, nx_to_range_representation, edge_array_to_range_representation



class Test1(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.G = nx.DiGraph()
        G = self.G
        G.add_nodes_from([0,1,2,3])
        G.add_edges_from([(0,2), (0,3), (1,2)])

    def test_nx_to_range(self):
        succ_range, succ_idx = nx_to_range_representation(self.G)
        np.testing.assert_array_equal(succ_range, [0,2,3,3,3])
        np.testing.assert_array_equal(succ_idx, [2,3,2])

    def test_SumAggregator_1(self):
        df = pd.DataFrame.from_dict({"A":[1,2,0,0]})
        df1 = apply_aggregator(SumAggregator, df, self.G)
        np.testing.assert_array_equal(df1["neigh_sum(A)"], [0,0,3,1])

    def test_SumAggregator_1_s(self):
        arr = pd.arrays.SparseArray.from_spmatrix(coo_array(([1,2], ([0,1], np.zeros(2),)), shape=(4, 1)))
        df_s = pd.DataFrame.from_dict({"A":arr})
        df1_s = apply_aggregator(SumAggregator, df_s, self.G)
        np.testing.assert_array_equal(df1_s["neigh_sum(A)"], [0,0,3,1])

    def test_SumMeanAggregator_1(self):
        df = pd.DataFrame.from_dict({"A":[1,2,0,0]})
        df1 = apply_aggregator(MeanAggregator, df, self.G)
        np.testing.assert_array_equal(df1["neigh_mean(A)"], [0,0,1.5,1])


    def test_MeanAggregator_1_s(self):
        arr = pd.arrays.SparseArray.from_spmatrix(coo_array(([1,2], ([0,1], np.zeros(2),)), shape=(4, 1)))
        df_s = pd.DataFrame.from_dict({"A":arr})
        df1_s = apply_aggregator(MeanAggregator, df_s, self.G)
        np.testing.assert_array_equal(df1_s["neigh_mean(A)"], [0,0,1.5,1])


    def test_edge_array_to_range_representation(self):
        succ_range, succ_idx = edge_array_to_range_representation(np.array([(0,1)],dtype=np.uint32),2)
        np.testing.assert_array_equal(succ_range, [0,1,1])
        np.testing.assert_array_equal(succ_idx, [1])

        succ_range, succ_idx = edge_array_to_range_representation(np.array([(0,1), (1,0)],dtype=np.uint32),2)
        np.testing.assert_array_equal(succ_range, [0,1,2])
        np.testing.assert_array_equal(succ_idx, [1,0])

        succ_range, succ_idx = edge_array_to_range_representation(np.array([(1,2), (2,3)],dtype=np.uint32),4)
        np.testing.assert_array_equal(succ_range, [0,0,1,2,2])
        np.testing.assert_array_equal(succ_idx, [2,3])


        succ_range, succ_idx = edge_array_to_range_representation(np.array([(1,2), (3,4)],dtype=np.uint32),5)
        np.testing.assert_array_equal(succ_range, [0,0,1,1,2,2])
        np.testing.assert_array_equal(succ_idx, [2,4])


if __name__ == "__main__":
    unittest.main()
