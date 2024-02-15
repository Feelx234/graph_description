import numpy as np
import pandas as pd
import networkx as nx
import pysubgroup as ps
from numba import njit
from scipy.sparse import coo_array

from collections import defaultdict


def get_range_representation(G, num_nodes_df):
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        assert num_nodes_df == G.number_of_nodes(), f"network and df don't have the same number of nodes {num_nodes_df} {G.number_of_nodes()}"
        return nx_to_range_representation(G)
    elif isinstance(G, np.ndarray):
        return edge_array_to_range_representation(G, num_nodes_df)
    else:
        raise NotImplementedError("Only networkx graph or numpy array supported")

@njit
def edge_array_to_range_representation(edge_arr, num_nodes):
    """Converts an edge array into a pred representation"""
    succ_range = np.empty(num_nodes+1, dtype=np.int32)
    succ_idx = np.empty(edge_arr.shape[0], dtype=np.int32)
    succ_range[0]=0
    order = np.argsort(edge_arr[:,0])
    edge_arr= edge_arr[order,:]
    succ_idx[:]=edge_arr[:,1]

    curr_node=0
    for num_edges_so_far in range(len(edge_arr)):
        u = edge_arr[num_edges_so_far,0]
        if curr_node==u:
            continue
        while curr_node < u:
            curr_node+=1
            succ_range[curr_node]=num_edges_so_far
    while curr_node < len(succ_range)-1:
        curr_node+=1
        succ_range[curr_node]=len(edge_arr)
    return succ_range, succ_idx


class SumAggregator:
    def __init__(self, column_names):
        self.column_names=column_names

    def apply(self, df, network):
        pred_range, pred_idx = get_range_representation(network, len(df))
        out_columns = {}
        for column_name in self.column_names:
            column = df[column_name]
            if isinstance(column.dtype, pd.SparseDtype):
                indices, values = propagate_sparse(column, pred_range, pred_idx)
                if len(indices)==0:
                    continue
                arr = pd.arrays.SparseArray.from_spmatrix(coo_array((values, (indices, np.zeros(len(indices)),)), shape=(len(column), 1)))
            else:
                arr = propagate_dense(column.to_numpy(), pred_range, pred_idx)
            out_columns["neigh_sum("+column_name+")"] = arr
        return pd.DataFrame.from_dict(out_columns)



def networkx_sparse_propagate(column, network):
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
    return indices, data


def nx_propagate_dense(column, network):
    arr = np.zeros(network.number_of_nodes())
    for node in network.nodes:
        for other in network.successors(node):
            arr[node] += column[other]



def nx_to_range_representation(G):
    """Converts a networkx Graph G into a pred representation"""
    pred_range = np.empty(G.number_of_nodes()+1, dtype=np.int32)
    num_edges = G.number_of_edges()
    if isinstance(G, nx.Graph):
        num_edges*=2
    pred_idx = np.empty(num_edges, dtype=np.int32)
    pred_range[0]=0
    start = 0

    if isinstance(G, nx.Graph):
        get_successors = G.neighbors
    elif isinstance(G, nx.DiGraph):
        get_successors = G.successors
    else:
        raise NotImplementedError(f"The following graph type is currently not supported: {type(G)}")
    for node in range(G.number_of_nodes()):
        l = list(get_successors(node))
        stop = start + len(l)
        pred_idx[start:stop]=l
        start = stop
        pred_range[node+1] = stop
    return pred_range, pred_idx


def propagate_sparse(column, pred_range, pred_idx):
    sp_index = column.array.sp_index.indices
    sp_values = column.array.sp_values
    #print("sparse", len(sp_index), len(sp_values))
    return _propagate_sparse(sp_index, sp_values, pred_range, pred_idx)


@njit
def _propagate_sparse(sp_index, sp_values, pred_range, pred_idx):
    dict_arr = {}
    dict_arr[0] = float(0.0)
    del dict_arr[0]
    for other, other_value in zip(sp_index, sp_values):
        start = pred_range[other]
        stop = pred_range[other+1]
        for id in range(start, stop):
            node = pred_idx[id]
            if node in dict_arr:
                dict_arr[node]+=other_value
            else:
                dict_arr[node]=other_value
    indices = np.empty(len(dict_arr), dtype=np.int32)
    values = np.empty(len(dict_arr), dtype=np.float64)
    i = 0
    for key, value in dict_arr.items():
        indices[i]=key
        values[i]=value
        i+=1
    return indices, values

def propagate_dense(values, pred_range, pred_idx):
    assert len(values) == len(pred_range)-1, (len(values), len(pred_range))
    return _propagate_dense(values, pred_range, pred_idx)

@njit
def _propagate_dense(values, pred_range, pred_idx):
    our_values = np.zeros(len(values), dtype=np.float64)

    for other, other_value in enumerate(values):
        start = pred_range[other]
        stop = pred_range[other+1]
        for id in range(start, stop):
            node = pred_idx[id]
            our_values[node]+=other_value
    return our_values



@njit
def get_out_degree(pred_range, pred_idx):
    out_degrees = np.bincount(pred_idx, minlength=len(pred_range)-1)
    assert len(out_degrees)==len(pred_range)-1
    return out_degrees


class MeanAggregator:
    def __init__(self, column_names):
        self.column_names=column_names

    def apply(self, df, network):
        pred_range, pred_idx = get_range_representation(network, len(df))
        out_degrees = np.array(get_out_degree(pred_range, pred_idx), dtype=np.float64)
        out_degrees = np.maximum(out_degrees, 1)
        out_columns = {}
        for column_name in self.column_names:
            column = df[column_name]
            if isinstance(column.dtype, pd.SparseDtype):
                indices, values = propagate_sparse(column, pred_range, pred_idx)
                if len(indices)==0:
                    continue
                values /= out_degrees[indices]
                arr = pd.arrays.SparseArray.from_spmatrix(coo_array((values, (indices, np.zeros(len(indices)),)), shape=(len(df), 1)))
            else:
                arr = propagate_dense(column.to_numpy(), pred_range, pred_idx)
                arr /= out_degrees
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
    #out_selectors = ps.create_selectors(out_df)
    return out_df#, out_selectors