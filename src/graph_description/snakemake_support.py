

from graph_description.networkx_aggregation import SumAggregator, MeanAggregator, apply_aggregator



def fix_column_name(name):
    return str(name).replace("<", " smaller ").replace("[", "{").replace("]",  "}")


def load_dataset_splitted(path_splits, path_df, return_train=True, return_val=True, return_test=False, return_full=False, labelsonly_dict=None, wildcards=None):
    import numpy as np
    splits = np.load(path_splits)
    import pandas as pd
    df  = pd.read_pickle(path_df)
    return _load_dataset_splitted(splits, df, return_train=return_train, return_val=return_val, return_test=return_test, return_full=return_full, labelsonly_dict=labelsonly_dict, wildcards=wildcards)

def _load_dataset_splitted(splits, df, return_train=True, return_val=True, return_test=False, return_full=False, labelsonly_dict=None, wildcards=None):
    if wildcards is not None:
        if "split_mutator" in wildcards.keys() and  "labelsonly" in wildcards.split_mutator:
            from graph_description.datasets import read_attributed_graph
            #try:
            G, df = read_attributed_graph(wildcards.dataset, kind="edges", group=wildcards.group)
            the_round = int(wildcards.round)
            assert the_round > 0, "for labelsonly, need a round > 1"
            labelsonly_dict=dict(output_path="some_prefix/split_labelsonly/other_stuff",
                                G=G,
                                round=the_round)
    if labelsonly_dict is not None:
        assert "round" in labelsonly_dict
        assert "G" in labelsonly_dict
        print("<<<<<< labelsonly >>>>>>")
        round_str = labelsonly_dict["round"]
        G = labelsonly_dict["G"]

        import pandas as pd
        round_int = int(round_str)#
        labels_copy = df["labels"].copy()
        tmp = df["labels"].copy()
        tmp[~splits["train_mask"]]=-1
        df = pd.DataFrame({"labels": tmp})
        df_labels = pd.get_dummies(df, columns=["labels"])
        dfs=[]
        for i in range(int(round_int)):
            df_labels = apply_aggregator((SumAggregator, MeanAggregator), df_labels, G)
            dfs.append(df_labels)
        total_df = pd.concat(dfs, axis=1)
        total_df.columns= list(map(fix_column_name, total_df.columns))
        df = total_df
        df["labels"] = labels_copy



    def get_by_split(split_name):
        mask = splits[split_name]
        mask_df = df[mask]
        X = mask_df.drop("labels", axis=1)
        y = mask_df["labels"]
        return X, y

    out = tuple()
    if return_train:
        out += get_by_split("train_mask")
    if return_val:
        out += get_by_split("val_mask")
    if return_test:
        out += get_by_split("test_mask")
    if return_full:
        out +=(df,)
    return out