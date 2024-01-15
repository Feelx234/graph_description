default_dir = "snakemake_base"
aggregated_datasets_dir = default_dir+"/aggregated_datasets"
splits_dir = default_dir+"/splits"
classifier_dir = default_dir+"/trained_classifiers"
scorer_dir = default_dir+"/scores"
experiment_dir = default_dir+"/experiments"

pickle_ending = ".pkl"
npz_ending = ".npz"
xgb_ending=".ubj"#".json"
txt_ending=".txt"
csv_ending=".csv"

def fix_column_name(name):
    return str(name).replace("<", " smaller ").replace("[", "{").replace("]",  "}")

import numpy as np

num_train_per_class_arr = [1,3,5,10,25,50,100,150,200]

wildcard_constraints:
    split_seed="\d+",
    num_train_per_class="\d+"

rule run_aggregation0:
    output : aggregated_datasets_dir+"/{dataset}_{group}_-1"+pickle_ending
    run:
        import pysubgroup as ps
        from graph_description.networkx_aggregation import SumAggregator, MeanAggregator, apply_aggregator
        from graph_description.datasets import read_attributed_graph
        group= wildcards.group


        G, df = read_attributed_graph(wildcards.dataset, kind="nx", group=wildcards.group)
        searchspace = ps.create_selectors(df, ignore=['labels'])
        searchspace = [sel for sel in searchspace if "==0" not in str(sel)]
        print(output)
        df.to_pickle(output[0])

rule run_aggregation:
    output : aggregated_datasets_dir+"/{dataset}_{group}_{round,[0-9]}"+pickle_ending
    run:
        import pandas as pd
        import numpy as np
        import pysubgroup as ps
        from graph_description.networkx_aggregation import SumAggregator, MeanAggregator, apply_aggregator
        from graph_description.datasets import read_attributed_graph

        G, df = read_attributed_graph(wildcards.dataset, kind="nx", group=wildcards.group)
        searchspace = ps.create_selectors(df, ignore=['labels'])
        searchspace = [sel for sel in searchspace if "==0" not in str(sel)]
        dfs = [df]
        df = pd.DataFrame.from_dict({str(selector): selector.covers(df) for selector in searchspace}|{"all_ones" : np.ones(len(df))})

        for i in range(int(wildcards.round)):
            df = apply_aggregator((SumAggregator, MeanAggregator), df, G)
            dfs.append(df)
        print(df.shape)
        total_df = pd.concat(dfs, axis=1)
        total_df.columns= list(map(fix_column_name, total_df.columns))
        total_df.to_pickle(output[0])


rule create_split:
    output : splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending
    run:
        import numpy as npxgb_ending
        from graph_description.datasets import read_attributed_graph, create_random_split
        G, df = read_attributed_graph(wildcards.dataset, kind="edges", group=wildcards.group)
        np.random.seed(int(wildcards.split_seed))
        if wildcards.num_test == "rest":
            num_test = "rest"
        else:
            num_test = int(num_test)
        train_mask, val_mask, test_mask = create_random_split(df, int(wildcards.num_train_per_class), int(wildcards.num_val), num_test)
        np.savez(output[0], train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


rule train_xgbclassifier:
    output :
        (classifier_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/xgbclass_{n_estimators}_{max_depth}_{clf_seed}"+xgb_ending)
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}"+pickle_ending
    run :
        import numpy as np
        import pandas as pd
        splits = np.load(input[0])
        train_mask = splits["train_mask"]

        df  = pd.read_pickle(input[1])
        train_df = df[train_mask]
        #print("number_of_columns", len(df.columns))
        X_train = train_df.drop("labels", axis=1)

        y_train = train_df["labels"]

        from xgboost import XGBClassifier
        bst = XGBClassifier(n_estimators=int(wildcards.n_estimators),
            max_depth=int(wildcards.max_depth),
            learning_rate=1,
            objective='binary:logistic',
            random_state=wildcards.clf_seed)
        bst.fit(X_train, y_train)
        bst.save_model(output[0])


rule test_xgbclassifier:
    input :
        (classifier_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/xgbclass_{n_estimators}_{max_depth}_{clf_seed}"+xgb_ending),
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}"+pickle_ending
    output :
        (scorer_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/score_{score_name}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/xgbclass_{n_estimators}_{max_depth}_{clf_seed}"+txt_ending),
    run :
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score
        splits = np.load(input[1])
        test_mask = splits["test_mask"]

        df  = pd.read_pickle(input[2])
        print("number_of_columns", len(df.columns))
        test_df = df[test_mask]
        X_test = test_df.drop("labels", axis=1)
        y_test = test_df["labels"]
        scorers = {
            "accuracy" : accuracy_score,
            "f1" : partial(f1_score, average="micro")
        }
        scorer = scorers[wildcards.score_name]

        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.load_model(input[0])
        y_test_predict = classifier.predict(X_test)
        score = scorer(y_test, y_test_predict)
        score = np.array([score])
        np.savetxt(output[0], score)
        print(list(wildcards.keys()))

def c_range(the_stop, start=1):
    vals = [(1,6),
            (2,20),
            (5,50),
            (10,100),
            (25,300),
            (50,600),
            (100,1000),
            (200,2000),
            (500,5000),
            (1000,10000),]
    if the_stop > 10_000:
        raise NotImplementedError("value to big")
    out = []
    for step, stop in vals:
        stop = min(stop, the_stop)
        if start >=stop:
            break
        out.extend(list(range(start, stop, step)))
        start=stop
    return out

from itertools import product

def agg_train_per_class_helper(wildcards):
    from graph_description.datasets import read_attributed_graph
    group= wildcards.group
    G, df = read_attributed_graph(wildcards.dataset, kind="edges", group=wildcards.group)
    max_num_train_for_network = np.bincount(df["labels"].to_numpy()).min()
    result = [(scorer_dir+
        f"/{wildcards.dataset}_{wildcards.group}"+
        f"/round_{wildcards.round}"+
        f"/score_{wildcards.score_name}"+
        f"/split_{num_train_per_class}_{wildcards.num_val}_{wildcards.num_test}_{split_seed}"+
        f"/xgbclass_{wildcards.n_estimators}_{wildcards.max_depth}_{wildcards.clf_seed}"+txt_ending)
        for num_train_per_class, split_seed in product(c_range(max_num_train_for_network), list(range(int(wildcards.max_split_seed))))]
    return result

rule agg_train_per_class:
    input :
        agg_train_per_class_helper
    output :
        (experiment_dir+
        "/agg_train_per_class"
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/score_{score_name}"+
        "/split_auto_{num_val}_{num_test}_{max_split_seed}"+
        "/xgbclass_{n_estimators}_{max_depth}_{clf_seed}"+csv_ending),
    run :
        import numpy as np
        import pandas as pd
        records = []
        #print(wildcards.keys())
        wildcard_values = []
        wildcard_keys = []
        for key, value in wildcards.items():
            if key =="input":
                continue
            else:
                wildcard_values.append(value)
                wildcard_keys.append(key)
                print(key,value)
        wildcard_values = tuple(wildcard_values)
        wildcard_keys = tuple(wildcard_keys)
        #print(wildcards.items())
        #wildcard_values = tuple(wildcards.values())
        #wildcard_keys = tuple(wildcards.keys())
        def extract_num_train_from_path(s):
            return int(s.split("split_")[1].split("/xgbclass")[0].split("_")[0])
        def extract_split_seed_from_path(s):
            return int(s.split("split_")[1].split("/xgbclass")[0].split("_")[3])

        for file_to_load in input:
            val = np.loadtxt(file_to_load)
            train_per_class = extract_num_train_from_path(file_to_load)
            split_seed = extract_split_seed_from_path(file_to_load)
            records.append(wildcard_values+(train_per_class, split_seed, file_to_load,val))
        df = pd.DataFrame.from_records(records, columns=wildcard_keys+("train_per_class", "split_seed", "path", "value"))
        df.to_csv(output[0], index=False)
# snakemake .\snakemake_base\experiments\agg_train_per_class\citeseer_None\round_1\score_accuracy\split_auto_0_rest_10\xgbclass_10_2_0.csv --cores 1 -f

# snakemake .\snakemake_base\experiments\agg_train_per_class\citeseer_None\round_1\score_accuracy\split_auto_0_rest_10\xgbclass_10_2_0.csv --cores 1 -f