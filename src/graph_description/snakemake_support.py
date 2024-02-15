

from graph_description.networkx_aggregation import SumAggregator, MeanAggregator, apply_aggregator



def fix_column_name(name):
    return str(name).replace("<", " smaller ").replace("[", "{").replace("]",  "}")


def load_dataset_splitted(path_splits, path_df, return_train=True, return_val=True, return_test=False, return_full=False, labelsonly_dict=None):
    import numpy as np
    splits = np.load(path_splits)
    import pandas as pd
    df  = pd.read_pickle(path_df)
    return load_dataset_splitted(splits, df, return_train=return_train, return_val=return_val, return_test=return_test, return_full=return_full, labelsonly_dict=labelsonly_dict)

def _load_dataset_splitted(splits, df, labelsonly_dict=None, return_train=True, return_val=True, return_test=False, return_full=False):
    output_path = None
    if labelsonly_dict is not None:
        assert "output_path" in labelsonly_dict
        assert "round" in labelsonly_dict
        assert "G" in labelsonly_dict

        output_path = labelsonly_dict["output_path"]
        round_str = labelsonly_dict["round"]
        G = labelsonly_dict["G"]
        
    if output_path is not None and "split_labelsonly" in output_path:
        import pandas as pd
        round_int = int(round_str)#
        tmp = df["labels"].copy()
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
        df["labels"] = tmp



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