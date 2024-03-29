from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_array
from graph_description.utils import get_dataset_folder


def is_in_ids(arr, ids):
    """Computes a bool array that is 1 if the corresponding entry in arr is in ids"""
    id_set = set(ids)
    out = np.empty(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        out[i] = v in id_set
    return out


def bring_network_and_attributes_in_line(df_edges, df_attributes, col1, col2):
    """Keeps only those node ids that are in both the edges and attributes"""
    edge_ids = np.union1d(df_edges[col1], df_edges[col2])
    valid_ids = np.intersect1d(df_attributes.index, edge_ids)

    df_attr_corr = df_attributes.loc[valid_ids]
    valid_edges_bools = np.logical_and(is_in_ids(df_edges[col1], valid_ids),
                                        is_in_ids(df_edges[col2], valid_ids))
    df_edges_corr = df_edges[valid_edges_bools]
    return df_edges_corr, df_attr_corr

def clean_covid_edges(df):
    df = df.reset_index(drop=True)
    to_add = []
    to_remove = []
    for i, (fro_, to) in enumerate(zip(df["infected_by"], df["id"])):
        try:
            if not pd.isna(fro_):
                int(fro_)
        except ValueError:
            to_remove.append(i)
            for u in fro_.split(","):
                to_add.append((int(u),to))
    df  = df.drop(to_remove, axis=0)
    new_df = pd.DataFrame.from_records(to_add, columns=df.columns)
    out_df = pd.concat((df, new_df)).dropna()
    out_df["infected_by"] = out_df["infected_by"].astype(np.int64)
    out_df = out_df.reset_index(drop=True)
    return out_df



def clean_covid(datasets_dir=None, write_files=False):
    """
    Take the raw PatientInfo.csv and convert into an attributed network


    """
    if datasets_dir is None:
        datasets_dir = Path("../datasets/").absolute()
    csv_path = (datasets_dir/"covid"/"PatientInfo.csv").resolve()
    print(csv_path)
    df = pd.read_csv(csv_path, header=0, sep=",", index_col="patient_id")

    df_out = df[["infected_by"]]
    df_out["id"] = df_out.index
    df_out = df_out.dropna()

    df_edges = clean_covid_edges(df_out)
    # find ids that are both present in the network and also have attributes
    df_edges_corr, df_attr_corr = bring_network_and_attributes_in_line(df_edges, df, "infected_by", "id")

    if write_files:
        df_edges_corr.to_csv("edges.csv", index=False)
        df_attr_corr.reset_index().to_csv("nodes.csv", index=False)

    return df_edges_corr, df_attr_corr



class Dataset:
    """Simple structure to store information on datasets"""
    def __init__(self, name, file_name, is_directed=False, delimiter=None, num_nodes=None):
        self.name=name
        self.file_name = file_name
        self.get_edges = self.get_edges_pandas
        self.skip_rows = 0
        self.is_directed=is_directed
        self.delimiter = delimiter
        self.requires_node_renaming=False
        self.num_nodes = num_nodes
        self.mapping=None


    def get_edges_pandas(self, datasets_dir):
        """Reads edges using pands read_csv function"""
        df = pd.read_csv(datasets_dir/self.file_name,
                         skiprows=self.skip_rows,
                         header=None,
                         sep=self.delimiter,
                         names=["from", "to"])
        edges = np.array([df["from"].to_numpy(), df["to"].to_numpy()],dtype=np.uint64).T

        if self.requires_node_renaming:
            edges, mapping =  relabel_edges(edges)
            self.mapping=mapping
        return edges

    def get_networkx(self, datasets_dir):
        edges = self.get_edges(datasets_dir.resolve())
        print(edges.ravel().max(), len(np.unique(edges.ravel())))
        return networkx_from_edges(edges, self.num_nodes, self.is_directed)


    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Dataset):
            return self.name==other.name
        else:
            raise ValueError()



def relabel_edges(edges):
    """relabels nodes such that they start from 0 consecutively"""
    unique = np.unique(edges.ravel())
    mapping = {key:val for key, val in zip(unique, range(len(unique)))}
    out_edges = np.empty_like(edges)
    for i,(e1,e2) in enumerate(edges):
        out_edges[i,0] = mapping[e1]
        out_edges[i,1] = mapping[e2]
    return out_edges, mapping




class AttributedDataset(Dataset):
    def __init__(self, *args, attributes_file=None, index_col=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes_file =  attributes_file
        self.index_col = index_col
        self.columns_to_drop=None
        self.group="other"
        self.identifier=form_identifier_from_group(self.name, self.group)


    def get_node_attributes(self, datasets_dir):
        df = pd.read_csv(datasets_dir/self.attributes_file, header=0, sep=self.delimiter, index_col=self.index_col)
        print(len(df))
        if self.requires_node_renaming:
            new_index = df.index.map(self.mapping)
            df = df[np.logical_not(pd.isna(new_index))]
            df.index = df.index.map(self.mapping)
        df.sort_index(inplace=True)
        columns_to_drop = [column for column in df.columns if np.all(pd.isna(df[column]))]
        if columns_to_drop:
            print("dropping all NAN columns", columns_to_drop)
            df.drop(columns_to_drop, axis=1, inplace=True)
        if self.columns_to_drop:
            df.drop(self.columns_to_drop, axis=1, inplace=True)

        return df




def npz_to_coo_array(data, prefix="attr"):
    """convert data from npz file into coo sparse array
    convert data from https://github.com/shchur/gnn-benchmark/tree/master/data
      into sparse matrix """
    assert prefix in ["adj", "attr"]
    def to_indices(data):
        n = len(data[prefix+"_indptr"]) -1
        return np.repeat(np.arange(n), np.diff(data[prefix+"_indptr"]))

    return coo_array((data[prefix+"_data"], (to_indices(data), data[prefix+"_indices"])), shape=data[prefix+"_shape"])

class NpzDataset:
    def __init__(self, name, file_name):
        self.name = name
        self.file_name = file_name
        self.data = None
        self.group="other"
        self.identifier=form_identifier_from_group(self.name, self.group)

    def ensure_data(self, datasets_dir):
        self.data = np.load(datasets_dir/self.file_name)

    def get_networkx(self, datasets_dir):
        self.ensure_data(datasets_dir)
        adjacency = npz_to_coo_array(self.data, "adj")
        return nx.from_scipy_sparse_array(adjacency, create_using=nx.DiGraph)


    def get_edges(self, datasets_dir):
        self.ensure_data(datasets_dir)
        adjacency = npz_to_coo_array(self.data, "adj")
        edges = np.vstack((adjacency.row, adjacency.col))
        if edges.shape[1]!=2:
            return edges.T
        return edges

    def get_node_attributes(self, datasets_dir):
        self.ensure_data(datasets_dir)
        attr_array = npz_to_coo_array(self.data, "attr")
        df = pd.DataFrame.sparse.from_spmatrix(attr_array)
        df["labels"] = self.data["labels"]
        return df


def form_identifier_from_group(name, group=None):
    return str(group).lower() + "_" + name.lower()


class TorchDataset:
    def __init__(self, name, group):
        self.name = name
        self.identifier = form_identifier_from_group(name, group)
        self.group = group.lower()
        self.data = None

    def get_networkx(self, datasets_dir):
        self.ensure_data(datasets_dir)
        edge_index = self.get_edges(datasets_dir)
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.data.y)))
        G.add_edges_from(edge_index)
        return G


    def get_edges(self, datasets_dir):
        self.ensure_data(datasets_dir)
        edge_index = self.data.edge_index
        if edge_index.shape[1]!=2:
            edge_index = edge_index.T
        return edge_index


    def get_node_attributes(self, datasets_dir):
        self.ensure_data(datasets_dir)
        if isinstance(self.data.x, coo_array):
            df = pd.DataFrame.sparse.from_spmatrix(self.data.x)
        else:
            df = pd.DataFrame(self.data.x)
        df["labels"] = self.data.y
        return df

    def ensure_data(self, datasets_dir):
        if self.data is None:
            self._load_data(datasets_dir)


    def _load_data(self, datasets_dir):
        from graph_description.torch_port.torch_datasets import Planetoid, CitationFull, WikiCS, Coauthor #pylint:disable=import-outside-toplevel
        group_mapping = {
            "planetoid" : Planetoid,
            "citation" : CitationFull,
            "citationfull" : CitationFull,
            "wikics" : WikiCS,
            "coauthor" : Coauthor
        }
        assert self.group in group_mapping
        group_class = group_mapping[self.group]
        if group_class is CitationFull:
            self.data = group_class(datasets_dir, self.name, to_undirected=False)._data #pylint:disable=protected-access
        else:
            self.data = group_class(datasets_dir, self.name)._data #pylint:disable=protected-access






deezer_data = AttributedDataset("deezer", "deezer/edges_directed.csv",
                                delimiter=",",
                                num_nodes=54573,
                                index_col="id",
                                attributes_file = "deezer/nodes.csv",
                               is_directed=True)
deezer_data.skip_rows = 1


covid_data = AttributedDataset("covid", "covid/edges.csv",
                                delimiter=",",
                                num_nodes=None,
                                index_col="patient_id",
                                attributes_file = "covid/nodes.csv",
                              is_directed=False)
covid_data.skip_rows = 1
covid_data.requires_node_renaming=True
covid_data.columns_to_drop = ["contact_number",
                              "symptom_onset_date",
                              "released_date",
                              "deceased_date",
                              "infected_by",
                              "country"]

citeseer_data = NpzDataset("citeseer", "citeseer.npz")
pubmed_data = NpzDataset("pubmed", "pubmed.npz")

planetoid_cora = TorchDataset("cora", "planetoid")
planetoid_citeseer = TorchDataset("citeseer", "planetoid")
planetoid_pubmed = TorchDataset("pubmed", "planetoid")
planetoid_datasets = [planetoid_pubmed, planetoid_cora, planetoid_citeseer]

citation_full_datasets = [
     TorchDataset("cora", "citationfull"),
     TorchDataset("cora_ml", "citationfull"),
     TorchDataset("citeseer", "citationfull"),
     TorchDataset("dblp", "citationfull"),
     TorchDataset("pubmed", "citationfull")
]

other_datasets = [
    TorchDataset("wikics", "wikics"),
    TorchDataset("cs", "coauthor"),
    TorchDataset("physics", "coauthor"),
]
# df_out = df[["infected_by"]]
# df_out["id"] = df_out.index
# df_out.dropna().to_csv("edges.csv", index=False)


all_datasets_list = ([deezer_data, covid_data, citeseer_data, pubmed_data] +
                     planetoid_datasets +
                     citation_full_datasets +
                     other_datasets)
all_datasets = {dataset.identifier : dataset for dataset in all_datasets_list}



def networkx_from_edges(edges, size, is_directed, is_multi=False):
    """Create a networkx graph from an edge list"""
    if size is None:
        unique = np.unique(edges.flatten())
        assert unique[0]==0, "expecting to start from 0 " + str(unique[:10])
        size = len(unique)
    if is_directed:
        if is_multi:
            G = nx.MultiDiGraph()
        else:
            G = nx.DiGraph()
    else:
        if is_multi:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
    G.add_nodes_from(list(range(size)))
    G.add_edges_from(edges)
    return G



def filter_min_component_size(G, df, min_component_size):

    large_components = list(comp for comp in nx.connected_components(G) if len(comp)>=min_component_size)

    sub_nodes = list(chain.from_iterable(large_components))
    G_out = G.subgraph(sub_nodes)
    df_out = df.loc[sub_nodes]
    return G_out, df_out



def choose_dataset(dataset_name, group=None):
    unique_identifier = form_identifier_from_group(dataset_name, group)
    if unique_identifier in all_datasets:
        return all_datasets[unique_identifier]

    if group is None or group=="auto":
        potential_datasets = [dataset for dataset in all_datasets.values() if dataset.name==dataset_name]
        if len(potential_datasets) == 1:
            return potential_datasets[0]
        elif len(potential_datasets) == 0:
            raise ValueError(f"Could not find a dataset with name {dataset_name}")
        else:
            dataset_names = [form_identifier_from_group(d.name, d.group) for d in potential_datasets]
            raise ValueError(f"There is no unique dataset with name {dataset_name}. Found {dataset_names}")
    raise ValueError(f"Could not find a dataset with name {dataset_name} and group {group} {list(all_datasets.keys())}")



def nx_read_attributed_graph(dataset_name, dataset_path=None, group=None):
    if dataset_path is None:
        dataset_path = get_dataset_folder()
    dataset = choose_dataset(dataset_name, group)

    G = dataset.get_networkx(dataset_path)
    df = dataset.get_node_attributes(dataset_path.resolve())
    return G, df



def edges_read_attributed_graph(dataset_name, dataset_path=None, group=None, split=None):
    if dataset_path is None:
        dataset_path = get_dataset_folder()
    dataset = choose_dataset(dataset_name, group)

    edges = dataset.get_edges(dataset_path)
    df = dataset.get_node_attributes(dataset_path.resolve())
    return edges, df



def create_random_split(df, num_train_per_class, num_val, num_test):
    y = df["labels"].to_numpy()
    num_classes = len(np.unique(y))
    num_nodes = len(df)
    train_mask = np.zeros(num_nodes, dtype=bool)

    for c in range(num_classes):
        idx = (y == c).nonzero()[0]
        idx = idx[np.random.permutation(idx.shape[0])[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero()[0]
    remaining = remaining[np.random.permutation(remaining.shape[0])]#pylint:disable=unsubscriptable-object, no-member

    val_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[remaining[:num_val]] = True

    test_mask = np.zeros(num_nodes, dtype=bool)
    if isinstance(num_test, str):
        assert num_test in ("remaining", "rest"), f"num_test is {num_test} which is not valid"
        test_mask[remaining[num_val:]] = True
    else:
        test_mask[remaining[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask



def read_attributed_graph(dataset_name, kind="edges", dataset_path=None, group=None, split=None):
    if dataset_path is None:
        dataset_path = get_dataset_folder()
    if group=="None":
        group = None
    dataset = choose_dataset(dataset_name, group)
    assert kind in ("edges", "nx", "networkx")

    if kind in ("edges",):
        graph = dataset.get_edges(dataset_path)
    elif kind in ("nx", "networkx"):
        graph = dataset.get_networkx(dataset_path)
    else:
        raise ValueError(f"Parameter kind is '{kind}' which is not supported.")
    df = dataset.get_node_attributes(dataset_path.resolve())

    if split is None:
        return graph, df
    elif split=="public":
        num_train_per_class, num_val, num_test = split
        splits = create_random_split(df, num_train_per_class, num_val, num_test)
        return graph, df, splits
    else:
        num_train_per_class, num_val, num_test = split
        splits = create_random_split(df, num_train_per_class, num_val, num_test)
        return graph, df, splits



def get_knecht_data(wave):
    """Return the adjacency and attributes of the Knecht dataset

    Note that wave 1 does not include the alcohol column and not all students are present in all datasets
    """


    folder = get_dataset_folder()/"klas12b"

    df = pd.read_csv(folder/"klas12b-demographics.dat", header=None, delimiter=" ")
    df.drop(0, axis=1, inplace=True)
    df.columns = ["sex", "age", "ethnicity", "father_religion"]
    df.sex = pd.Categorical.from_codes(df.sex-1, ["female", "male"])
    df.ethnicity = pd.Categorical.from_codes(df.ethnicity-1, ["Dutch", "non-Dutch"])
    df.father_religion = pd.Categorical.from_codes(df.father_religion-1, ["Christian", "non-religious", "other religion"])
    df_out = df

    # advice
    df = pd.read_csv(folder/"klas12b-advice.dat", header=None)
    df.columns= ["advice"]

    df[df.advice == 0]=np.nan
    df_out["advice"] = df.advice

    # alcohol
    if wave > 1:
        df = pd.read_csv(folder/"klas12b-alcohol.dat", header=None, delimiter=" ")
        df.drop(0, axis=1, inplace=True)
        alcohol_column = df[wave-1]-1
        categories = ["never", "once", "2-4 times", "5-10 times", "more than 10 times"]
        alcohol_column = pd.Categorical.from_codes(alcohol_column, categories=categories, ordered=True)
        df_out["alcohol"] = alcohol_column

    # delinquency
    df = pd.read_csv(folder/"klas12b-delinquency.dat", header=None, delimiter=" ")
    df.drop(0, axis=1, inplace=True)

    delin_column = df[wave]-1
    categories = ["never", "once", "2-4 times", "5-10 times", "more than 10 times"]
    delin_column = pd.Categorical.from_codes(delin_column, categories=categories, ordered=True)
    df_out["delinquency"] = delin_column

    df_out["identifier"] = df_out.index

    adj = np.loadtxt(folder/f"klas12b-net-{wave}.dat")
    correct = adj<=1
    np.logical_and(np.all(correct, axis=0), np.all(correct.T, axis=0))
    rows_correct = np.any(correct, axis=1)
    adj = adj[rows_correct, :]
    adj = adj[:, rows_correct]
    df_out = df_out[rows_correct]

    df_out.reset_index(drop=True, inplace=True)


    return adj, df_out