import numpy as np
import pandas as pd
import networkx as nx
import pysubgroup as ps
from scipy.sparse import coo_array

from collections import defaultdict

class SumAggregator:
    def __init__(self, column_names):
        self.column_names=column_names

    def apply(self, df, network):
        out_columns = {}
        for column_name in self.column_names:
            column = df[column_name]
            if isinstance(column.dtype, pd.SparseDtype):
                dict_arr = defaultdict(float)
                sp_index = column.array.sp_index.indices
                sp_values = column.array.sp_values
                for other, other_value in zip(sp_index, sp_values):
                    for node in network.predecessors(other):
                        dict_arr[node]+=other_value
                data = np.empty(len(dict_arr))
                indices = np.empty(len(dict_arr))
                for i, (key, value) in enumerate(dict_arr.items()):
                    data[i] = value
                    indices[i] = key
                arr = pd.arrays.SparseArray.from_spmatrix(coo_array((data, (indices, np.zeros(len(indices)),)), shape=(len(df), 1)))
            else:
                arr = np.zeros(network.number_of_nodes())
                for node in network.nodes:
                    for other in network.successors(node):
                        arr[node] += column[other]
            out_columns["neigh_sum("+column_name+")"] = arr
        return pd.DataFrame.from_dict(out_columns)



class MeanAggregator:
    def __init__(self, column_names):
        self.column_names=column_names

    def apply(self, df, network):
        out_columns = {}
        for column_name in self.column_names:
            column = df[column_name]
            if isinstance(column.dtype, pd.SparseDtype):
                dict_arr = defaultdict(float)
                sp_index = column.array.sp_index.indices
                sp_values = column.array.sp_values
                for other, other_value in zip(sp_index, sp_values):
                    for node in network.predecessors(other):
                        dict_arr[node]+=other_value
                data = np.empty(len(dict_arr))
                indices = np.empty(len(dict_arr))
                for i, (key, value) in enumerate(dict_arr.items()):
                    data[i] = value / network.out_degree(key)
                    indices[i] = key
                arr = pd.arrays.SparseArray.from_spmatrix(coo_array((data, (indices, np.zeros(len(indices)),)), shape=(len(df), 1)))
            else:
                arr = np.zeros(network.number_of_nodes())
                for node in network.nodes:
                    for other in network.successors(node):
                        arr[node] += column[other]
                for node in network.nodes:
                    arr[node]/= network.out_degree(node)
            out_columns["neigh_mean("+column_name+")"] = arr
        return pd.DataFrame.from_dict(out_columns)



def apply_aggregator(aggregator_classes, data, network, selectors=None):
    print("init")
    if not selectors is None:
        data_df = pd.DataFrame.from_dict({str(selector): selector.covers(data) for selector in selectors}|{"all_ones" : np.ones(len(data))})
    else:
        data_df = data
    print("prep done")
    if not isinstance(aggregator_classes, (list, tuple)):
        aggregator_classes = [aggregator_classes]
    result_data = []
    for aggregator_class in aggregator_classes:
        agg = aggregator_class(data_df.columns)
        agg_df = agg.apply(data_df, network)
        result_data.append(agg_df)

    out_df = pd.concat(result_data, axis=1)
    out_selectors = ps.create_selectors(out_df)
    #if not selectors is None:
    #    out_selectors.extend(selectors)
    
    return out_df, out_selectors