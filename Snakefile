# type: ignore
default_dir = "snakemake_base"
aggregated_datasets_dir = default_dir+"/aggregated_datasets"
splits_dir = default_dir+"/splits"
classifier_dir = default_dir+"/trained_classifiers"
prediction_dir = default_dir+"/classifier_predictions"
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
    num_train_per_class="auto|\d+|max_\d+",
    early_stopping_rounds=".*|_\d+"


rule test_range:
    output : aggregated_datasets_dir+"/{test}.txt"
    run:
        print(list(eval(wildcards.test)))

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
        #print(output)
        df.to_pickle(output[0])




rule run_aggregation:
    output :
        aggregated_datasets_dir+"/{dataset}_{group}_{round,[0-9]}"+pickle_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round,[0-9]}_dense"+pickle_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round,[0-9]}_labels.npy"
    run:
        import pandas as pd
        import numpy as np
        import pysubgroup as ps
        from graph_description.networkx_aggregation import SumAggregator, MeanAggregator, apply_aggregator
        from graph_description.datasets import read_attributed_graph
        #try:
        G, df = read_attributed_graph(wildcards.dataset, kind="nx", group=wildcards.group)
        #except ValueError:
        #    G, df = read_attributed_graph(wildcards.dataset, kind="nx", group="other")
        searchspace = ps.create_selectors(df, ignore=['labels'])
        searchspace = [sel for sel in searchspace if "==0" not in str(sel)]
        dfs = [df]
        df = pd.DataFrame.from_dict({str(selector): selector.covers(df) for selector in searchspace}|{"all_ones" : np.ones(len(df))})

        for i in range(int(wildcards.round)):
            df = apply_aggregator((SumAggregator, MeanAggregator), df, G)
            dfs.append(df)
        #print(df.shape)
        total_df = pd.concat(dfs, axis=1)
        total_df.columns= list(map(fix_column_name, total_df.columns))
        total_df.to_pickle(output[0])
        print(total_df.shape)

        from sklearn.utils.validation import check_array
        tmp = check_array(total_df)
        print(tmp.shape)
        df_dense = pd.DataFrame(tmp, columns = total_df.columns)
        df_dense.to_pickle(output[1])

        arr = total_df["labels"].to_numpy()
        np.save(output[2], arr, allow_pickle=False)




rule create_split:
    output : splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending
    run:
        import numpy as npxgb_ending
        from graph_description.datasets import read_attributed_graph, create_random_split
        G, df = read_attributed_graph(wildcards.dataset, kind="edges", group=wildcards.group)
        print(np.max(G), print(df.shape))
        np.random.seed(int(wildcards.split_seed))
        if wildcards.num_test == "rest":
            num_test = "rest"
        else:
            num_test = int(num_test)
        train_mask, val_mask, test_mask = create_random_split(df, int(wildcards.num_train_per_class), int(wildcards.num_val), num_test)
        np.savez(output[0], train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

from sklearn.metrics import accuracy_score
def my_accuracy(y_true, y_pred):
    if len(y_pred.shape)>1:
        true_labels = np.argmax(y_pred, axis=1)
    else:
        true_labels=y_pred

    return 1-accuracy_score(y_true, true_labels)

xgb_raw = "/xgbclass_{n_estimators,[^_]+}_{max_depth,[^_]+}_{clf_seed,[^_]+}{early_stopping_rounds,.*}"
xgb_str = xgb_raw+xgb_ending
rule train_xgbclassifier:
    output :
        (classifier_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        xgb_str)
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}"+pickle_ending
    threads :
        4
    run :
        import numpy as np
        import pandas as pd

        splits = np.load(input[0])
        train_mask = splits["train_mask"]
        val_mask = splits["val_mask"]

        df  = pd.read_pickle(input[1])
        train_df = df[train_mask]
        #print("number_of_columns", len(df.columns))
        X_train = train_df.drop("labels", axis=1)

        y_train = train_df["labels"]

        from xgboost import XGBClassifier
        #print({key:wildcards[key] for key in wildcards.keys()})
        if len(wildcards.early_stopping_rounds)>0:
            if val_mask.sum()!=0:
                print("TRAIN CLASSIFIER: USING EVAL SET")

                from xgboost.callback import TrainingCallback
                class ChildWeightScheduler(TrainingCallback):
                    def __init__(self, gamma, timespan=10, min_size=1):
                        self.gamma=int(gamma)
                        self.timespan=timespan
                        self.min_size=min_size
                    def after_iteration(self, model, epoch, evals_log):
                        #print(model.attributes())
                        if epoch>0 and epoch % self.timespan == 0:
                            self.gamma=max(self.min_size, self.gamma-1)
                        model.set_param("min_child_weight", self.gamma)
                        #print(dir(model))
                        #model.gamma = model.gamma*0.95
                        #print(evals_log)


                val_df = df[val_mask]
                X_val = val_df.drop("labels", axis=1)
                y_val = val_df["labels"]
                train_per_class = int(wildcards.num_train_per_class)
                if train_per_class <= 10:
                    sample = 1
                elif train_per_class>=100:
                    sample=0.3
                else:
                    sample=0.5

                child_scheduler = ChildWeightScheduler(max(min(sample*train_per_class,10),1),10,1)
                params = dict(
                    n_estimators=int(wildcards.n_estimators),
                    max_depth=child_scheduler.gamma,
                    learning_rate=0.03,
                    objective='multi:softmax',
                    random_state=wildcards.clf_seed,
                    eval_metric=my_accuracy,
                    disable_default_eval_metric= 1,
                    n_jobs=1,
                    early_stopping_rounds=int(wildcards.early_stopping_rounds[1:]),
                    min_child_weight=5,
                #    tree_method="approx",
                #    gamma=1,
                    subsample=sample,
                    colsample_bytree = sample,
                    #max_delta_step=0.2
                #    reg_lambda = 50,
                #    reg_alpha = 1.5,
                    colsample_bylevel = sample,
                    colsample_bynode = sample,
                    callbacks = (child_scheduler,)
                )

                bst = XGBClassifier(**params)
                result = bst.fit(X_train,
                        y_train,
                        eval_set=[(X_train, y_train), (X_val, y_val)],
                        verbose=False)
                #print(dir(bst))
                #print("EVAL RESULTS", bst.evals_result())
                print("Best_iteration", bst.best_iteration, "best_score", bst.best_score)

            else:
                raise ValueError("Early stopping provided but validation set is empty!")
        else:
            print("TRAIN CLASSIFIER: NORMAL TRAINING")
            bst = XGBClassifier(n_estimators=int(wildcards.n_estimators),
                max_depth=int(wildcards.max_depth),
                learning_rate=1,
                objective='multi:softmax',
                random_state=wildcards.clf_seed,
                n_jobs=threads)
            bst.fit(X_train, y_train)

        bst.save_model(output[0])







imodels_ending = ".npy"
imodels_raw = "/rulefit_{max_rules}"
imodels_str = imodels_raw+imodels_ending
from pathlib import Path
rule get_rulefit_predictions:
    output :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        imodels_raw+".npy")
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}_dense"+pickle_ending
    run :
        from graph_description.gnn.run import main
        from imodels import RuleFitClassifier
        from sklearn.multiclass import OneVsRestClassifier

        import numpy as np
        import pandas as pd
        splits = np.load(input[0])
        train_mask = splits["train_mask"]
        val_mask = splits["val_mask"]

        df  = pd.read_pickle(input[1])
        train_df = df[train_mask]
        #print("number_of_columns", len(df.columns))
        X_train = train_df.drop("labels", axis=1)

        y_train = train_df["labels"]

        clf = OneVsRestClassifier(RuleFitClassifier(max_rules=int(wildcards.max_rules), cv=False))
        #try:
        clf.fit(X_train, y_train)
        prediction = clf.predict(df.drop("labels", axis=1))
        np.save(output[0], prediction, allow_pickle=False)
        #except RuleException:
        #    np.save(output[0], np.zeros(len(df),dtype=int), allow_pickle=False)









#print("WORKFLOW", )
gnn_ending = ".npy"
gnn_raw = "/{gnn_kind}_{init_seed}_{train_seed}"
gnn_str = gnn_raw+gnn_ending
from pathlib import Path
rule get_gnn_predictions:
    output :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/baselines"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        gnn_str)
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending
    run :
        from graph_description.gnn.run import main

        import numpy as np
        import pandas as pd
        splits = np.load(input[0])
        splits = {"train" : splits["train_mask"],
             "valid" : splits["val_mask"],
             "test" : splits["test_mask"]}

        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        config_dir = Path(workflow.snakefile).parent/"src"/"graph_description"/"gnn"/"config"
        print(config_dir)
        data_root = Path(workflow.snakefile).parent/"pytorch_datasets"
        with initialize_config_dir(config_dir=str(config_dir), job_name="test_app"):
            cfg = compose(config_name="main",
                          overrides=["cuda=0",
                                     f"model={wildcards.gnn_kind}",
                                     f"dataset={wildcards.dataset}",
                                     f"data_root={data_root}"])

            #print(OmegaConf.to_yaml(cfg))
            prediction = main(cfg, splits, wildcards.init_seed, wildcards.train_seed)
            np.save(output[0], prediction, allow_pickle=False)

# snakemake "./snakemake_base/classifier_predictions/pubmed_planetoid/baselines/split_20_500_rest_1/gat2017_0_0.npy" --cores 1


def process_score_name(score_name):
    from sklearn.metrics import accuracy_score, f1_score
    if "_" in score_name:
        score_name, set_name = score_name.split("_")
        assert set_name in ("test", "val", "train")
    else:
        set_name="test"
    scorers = {
            "accuracy" : accuracy_score,
            "f1" : partial(f1_score, average="micro")
    }
    return scorers[score_name], set_name

rule test_gnnclassifier:
    input :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/{joker2}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/{joker}.npy"),
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_0"+pickle_ending
    output :
        (scorer_dir+
        "/{dataset}_{group}"+
        "/{joker2}"+
        "/score_{score_name}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/{joker}"+txt_ending),
    run :
        import numpy as np
        import pandas as pd

        scorer, split_name = process_score_name(wildcards.score_name)
        splits = np.load(input[1])
        mask = splits[split_name+"_mask"]

        df  = pd.read_pickle(input[2])
        test_df = df[mask]
        y_test = test_df["labels"]

        y_test_predict = np.load(input[0])[mask]
        score = scorer(y_test, y_test_predict)
        score = np.array([score])
        np.savetxt(output[0], score)
# snakemake "./snakemake_base/scores/citeseer_planetoid/baselines/score_accuracy/split_20_500_rest_1/gat2017_0_0.txt" --cores 1




rule test_xgbclassifier:
    input :
        (classifier_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        xgb_str),
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}"+pickle_ending
    output :
        (scorer_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/score_{score_name}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        xgb_raw+txt_ending),
    run :
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score
        splits = np.load(input[1])
        scorer, split_name = process_score_name(wildcards.score_name)
        test_mask = splits[split_name+"_mask"]

        df  = pd.read_pickle(input[2])
        #print("number_of_columns", len(df.columns))
        test_df = df[test_mask]
        X_test = test_df.drop("labels", axis=1)
        y_test = test_df["labels"]

        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.load_model(input[0])
        y_test_predict = classifier.predict(X_test)
        score = scorer(y_test, y_test_predict)
        score = np.array([score])
        np.savetxt(output[0], score)
        #print(list(wildcards.keys()))

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
        if start >stop:
            break
        out.extend(list(range(start, stop, step)))
        start=stop
    return out

crange=c_range

from itertools import product

def effify(non_f_str: str):
    f_string =  f'f"""{non_f_str}"""'
    #print(f_string)
    return f_string


def unpack_wildcards(wildcards, _locals):
    for key in wildcards.keys():
        print(key)
        if key.startswith("dyn_"):
            value = eval(wildcards[key])
            if isinstance(value, (list, tuple, range)):
                _locals[key]=value
            else:
                _locals[key]=(value,)
        else:
            _locals[key]=wildcards[key]
    #print(_locals)

def agg_train_per_class_helper(wildcards):
    from graph_description.datasets import read_attributed_graph
    group= wildcards.group
    #G, df = read_attributed_graph(wildcards.dataset, kind="edges", group=wildcards.group)
    #max_num_train_for_network = 200#np.bincount(df["labels"].to_numpy()).min()
    print("AAAA")
    unpack_wildcards(wildcards, globals())
    #print(globals())
    #print(n_estimators)
    result = [(scorer_dir+
        f"/{wildcards.dataset}_{wildcards.group}"+
        f"/{wildcards.round}"+
        f"/score_{wildcards.score_name}"+
        f"/split_{num_train_per_class}_{wildcards.num_val}_{wildcards.num_test}_{split_seed}"+
        wildcards.joker+txt_ending)
        for num_train_per_class, split_seed in product(dyn_num_train_per_class, dyn_split_seed)]
    print(result)
    return result

    # message: "rule agg_train_per_class\r\n\toutput: {output}\r\n\twildcards: {wildcards}."

rule agg_train_per_class:
    input :
        agg_train_per_class_helper
    output :
        (experiment_dir+
        "/agg_train_per_class"
        "/{dataset}_{group}"+
        "/{round,baselines|round_[0-9]+}"+
        "/score_{score_name}"+
        "/split_{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
        "{joker,.*}"+csv_ending),

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
                #print(key,value)
        wildcard_values = tuple(wildcard_values)
        wildcard_keys = tuple(wildcard_keys)
        #print(wildcards.items())
        #wildcard_values = tuple(wildcards.values())
        #wildcard_keys = tuple(wildcards.keys())
        def extract_num_train_from_path(s):
            if "/xgbclass" in s:
                return int(s.split("split_")[1].split("/xgbclass")[0].split("_")[0])
            return 0
        def extract_split_seed_from_path(s):
            if "/xgbclass" in s:
                return int(s.split("split_")[1].split("/xgbclass")[0].split("_")[3])
            return 0

        for file_to_load in input:
            val = np.loadtxt(file_to_load)
            train_per_class = extract_num_train_from_path(file_to_load)
            split_seed = extract_split_seed_from_path(file_to_load)
            records.append(wildcard_values+(train_per_class, split_seed, file_to_load,val))
        df = pd.DataFrame.from_records(records, columns=wildcard_keys+("train_per_class", "split_seed", "path", "value"))
        df.to_csv(output[0], index=False)
# snakemake .\snakemake_base\experiments\agg_train_per_class\citeseer_None\round_1\score_accuracy\split_auto_0_rest_10\xgbclass_10_2_0.csv --cores 1 -f

# snakemake .\snakemake_base\experiments\agg_train_per_class\citeseer_None\round_1\score_accuracy\split_auto_0_rest_10\xgbclass_10_2_0.csv --cores 1 -f



# snakemake ".\snakemake_base\experiments\agg_train_per_class\pubmed_planetoid\round_0\score_accuracy\split_1_0_rest_0\xgbclass_10_2_0.csv"

# snakemake ./snakemake_base/experiments/agg_train_per_class/pubmed_planetoid/round_0/score_accuracy/split_20_500_rest_0/xgbclass_10_2_0.csv --cores 1

#  snakemake "./snakemake_base/experiments/agg_train_per_class/pubmed_planetoid/round_0/score_accuracy/split_crange(200)_500_rest_0/xgbclass_10_2_0_5.csv" --cores 16


# snakemake "./snakemake_base/experiments/agg_train_per_class/pubmed_planetoid/baselines/score_accuracy/split_crange(200)_500_rest_0/gcn2017_0_0.csv" --cores 16 -f -R