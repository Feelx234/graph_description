from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

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
        print(df["from"])
        edges = np.array([df["from"].to_numpy(), df["to"].to_numpy()],dtype=np.uint64).T

        if self.requires_node_renaming:
            edges, mapping =  relabel_edges(edges)
            self.mapping=mapping
        return edges


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
# df_out = df[["infected_by"]]
# df_out["id"] = df_out.index
# df_out.dropna().to_csv("edges.csv", index=False)


all_datasets_list = [deezer_data, covid_data]
all_datasets = {dataset.name : dataset for dataset in all_datasets_list}



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



def nx_read_attributed_graph(dataset_name, dataset_path=None):
    if dataset_path is None:
        dataset_path = Path("../datasets/").absolute()
    dataset = all_datasets[dataset_name]

    edges = dataset.get_edges(dataset_path.resolve())
    print(edges.ravel().max(), len(np.unique(edges.ravel())))
    G = networkx_from_edges(edges, dataset.num_nodes, dataset.is_directed)
    df = dataset.get_node_attributes(dataset_path.resolve())
    return G, df





def get_knecht_data(wave):
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
        alcohol_column = pd.Categorical.from_codes(alcohol_column, categories = ["never", "once", "2-4 times", "5-10 times", "more than 10 times"], ordered=True)
        df_out["alcohol"] = alcohol_column

    # delinquency
    df = pd.read_csv(folder/"klas12b-delinquency.dat", header=None, delimiter=" ")
    df.drop(0, axis=1, inplace=True)

    delin_column = df[wave]-1
    delin_column = pd.Categorical.from_codes(delin_column, categories = ["never", "once", "2-4 times", "5-10 times", "more than 10 times"], ordered=True)
    df_out["delinquency"] = delin_column

    adj = np.loadtxt(folder/f"klas12b-net-{wave}.dat")
    correct = adj<=1
    np.logical_and(np.all(correct, axis=0), np.all(correct.T, axis=0))
    rows_correct = np.any(correct, axis=1)
    adj = adj[rows_correct, :]
    adj = adj[:, rows_correct]
    df_out = df_out[rows_correct]

    return adj, df_out