# type: ignore
default_dir = "snakemake_base"
aggregated_datasets_dir = default_dir+"/aggregated_datasets"
splits_dir = default_dir+"/splits"
classifier_dir = default_dir+"/trained_classifiers"
prediction_dir = default_dir+"/classifier_predictions"
params_dir = default_dir+"/optimal_params"
scorer_dir = default_dir+"/scores"
experiment_dir = default_dir+"/experiments"

pickle_ending = ".pkl"
npz_ending = ".npz"
xgb_ending=".ubj"#".json"
txt_ending=".txt"
csv_ending=".csv"

from functools import partial

from graph_description.snakemake_support import *

import numpy as np


wildcard_constraints:
    split_seed="\d+",
    num_train_per_class="auto|\d+|max_\d+",
    early_stopping_rounds=".*|_\d+"


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
        #print(np.max(G), print(df.shape))
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

xgb_raw_no_constr = "/xgbclass_{n_estimators}_{max_depth}_{clf_seed}{early_stopping_rounds}"
xgb_str_no_constr = xgb_raw_no_constr+xgb_ending
rule xgb_train_classifier:
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


def load_xgb(splits_path, df_path, wildcards=None):
    import numpy as np
    X_train, y_train, X_val, y_val = load_dataset_splitted(splits_path, df_path, wildcards=wildcards)
    num_classes = len(np.bincount(y_train))

    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
    dval = xgb.DMatrix(X_val, label=y_val, missing=np.NaN)
    return dtrain, dval, num_classes

def run_and_save_study(objective, study_name, direction, n_trials, out_path):

    import optuna
    storage_path = Path(out_path).parent/"journal.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(storage_path)),
    )

    study = optuna.create_study(
        storage=storage,  # Specify the storage URL here.
        study_name=study_name,
        load_if_exists=True,
        direction=direction
    )

    n_trials = int(n_trials)
    num_remaining_trials = n_trials-len(study.get_trials())
    if  num_remaining_trials > 0:
        study.optimize(objective, n_trials=num_remaining_trials)

    import json
    with open(out_path, 'w') as file:
        file.write(json.dumps(study.best_params, indent=4))




rule xgb_optimize_optuna:
    output:
        (params_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split{split_mutator,_|_labelsonly_}{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/params_xgb_{n_trials}.json")
    input:
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}"+pickle_ending
    run:
        dtrain, dval, num_classes = load_xgb(input[0], input[1], wildcards=wildcards)

        from graph_description.training_utils import xgb_objective

        objective = partial(xgb_objective,
                            num_classes=num_classes,
                            dtrain=dtrain,
                            dval=dval)

        run_and_save_study(
            objective=objective,
            study_name = f"{wildcards.dataset}-{wildcards.round}-{wildcards.num_train_per_class}-xgb",
            direction="minimize",
            n_trials=wildcards.n_trials,
            out_path=output[0]
        )
# snakemake ./snakemake_base/optimal_params/citeseer_planetoid/round_0/split_20_500_rest_0/params_xgb_10.json --cores 1

full_output_split_str =  "/split{split_mutator,_|_labelsonly_}{num_train_per_class}_{num_val}_{num_test}_{split_seed}"
full_input_split_str =  "/split{split_mutator}{num_train_per_class}_{num_val}_{num_test}_{split_seed}"

rule xgb_train_predict_opticlassifier:
    output :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        full_output_split_str+
        "/xgbclass_opti({opti_split_seed},{param_search_n_trials}).npy")
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}"+pickle_ending,
        (params_dir+
            "/{dataset}_{group}"+
            "/round_{round}"+
            "/split{split_mutator}{num_train_per_class}_{num_val}_{num_test}_{opti_split_seed}"+
            "/params_xgb_{param_search_n_trials}.json")
    run :
        import pandas as pd
        dtrain, dval, num_classes = load_xgb(input[0], input[1])
        df  = pd.read_pickle(input[1])

        from graph_description.training_utils import xgb_get_config, TrialWrapperDict
        from xgboost import XGBClassifier

        trial = TrialWrapperDict(input[2])
        config=xgb_get_config(trial, num_classes)
        bst = XGBClassifier(**config)


        result = bst.fit(dtrain.get_data(),dtrain.get_label(),
            eval_set=[(dval.get_data(),dval.get_label())],
            verbose=False)

        prediction = bst.predict(df.drop("labels", axis=1))
        np.save(output[0], prediction, allow_pickle=False)
# snakemake "./snakemake_base/classifier_predictions/citeseer_planetoid/round_0/split_20_500_rest_0/xgbclass_opti(0,100).npy" --cores 1




rule gnn_optimize_optuna:
    output:
        (params_dir+
        "/{dataset}_{group}"+
        "/baselines"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"
        "/params_{gnn_kind,gcn2017|gat2017}_{n_trials}.json")
    input:
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_0_labels.npy"
    run:

        splits = np.load(input[0])
        splits = {"train" : splits["train_mask"],
            "valid" : splits["val_mask"],
            "test" : splits["test_mask"]}
        train_mask = splits["train"]
        val_mask = splits["valid"]

        labels  = np.load(input[1])
        y_val = labels[val_mask]


        from graph_description.training_utils import gnn_objective

        objective = partial(gnn_objective,
                            gnn_kind=wildcards.gnn_kind,
                            dataset=wildcards.dataset,
                            group=wildcards.group,
                            splits=splits, y_val=y_val)

        run_and_save_study(
            objective=objective,
            study_name = f"{wildcards.dataset}-X-{wildcards.num_train_per_class}-{wildcards.gnn_kind}",
            direction="maximize",
            n_trials=wildcards.n_trials,
            out_path=output[0]
        )

# snakemake ./snakemake_base/optimal_params/citeseer_planetoid/round_0/split_20_500_rest_0/params_gat2017_10.json --cores 1



imodels_ending = ".npy"
imodels_raw = "/rulefit_{max_rules}"
imodels_str = imodels_raw+imodels_ending
from pathlib import Path
rule rulefit_train_predict:
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

        (X_train, y_train, df)=load_dataset_splitted(input[0], input[1], return_val=False, return_full=True)

        clf = OneVsRestClassifier(RuleFitClassifier(max_rules=int(wildcards.max_rules), cv=False))

        clf.fit(X_train, y_train)
        prediction = clf.predict(df.drop("labels", axis=1))
        np.save(output[0], prediction, allow_pickle=False)












rule rulefit_optimize_optuna:
    output:
        (params_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}/params_rulefit{max_rules}_{n_trials}.json")
    input:
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}_dense"+pickle_ending
    run:
        (X_train, y_train, X_val, y_val)=load_dataset_splitted(input[0], input[1], output[0], round=wildcards.round)

        from graph_description.training_utils import rulefit_objective

        objective = partial(rulefit_objective,
                            max_rules=wildcards.max_rules,
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val)

        run_and_save_study(
            objective=objective,
            study_name = f"{wildcards.dataset}-{wildcards.round}-{wildcards.num_train_per_class}-rulefit{wildcards.max_rules}",
            direction="maximize",
            n_trials=wildcards.n_trials,
            out_path=output[0]
        )
# snakemake ./snakemake_base/optimal_params/citeseer_planetoid/round_0/split_20_500_rest_0/params_xgb_10.json --cores 1



rule rulefit_train_predict_opticlassifier:
    output :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/rulefit{max_rules,[0-9]+}_opti({opti_split_seed},{param_search_n_trials}).npy")
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}_dense"+pickle_ending,
        (params_dir+
            "/{dataset}_{group}"+
            "/round_{round}"+
            "/split_{num_train_per_class}_{num_val}_{num_test}_{opti_split_seed}"+
            "/params_rulefit{max_rules}_{param_search_n_trials}.json")
    run :
        (X_train, y_train, df)=load_dataset_splitted(input[0], input[1], return_val=False, return_full=True)

        from graph_description.training_utils import rulefit_train_classifier, TrialWrapperDict

        trial = TrialWrapperDict(input[2])
        clf = rulefit_train_classifier(trial, int(wildcards.max_rules), X_train, y_train)

        prediction = clf.predict(df.drop("labels", axis=1))
        np.save(output[0], prediction, allow_pickle=False)


def create_joined_train_val_set_for_sgdclassifier(X_train, y_train, X_val, y_val):
    import pandas as pd
    X_train_val = pd.concat((X_train, X_val), axis=0)
    y_train_val  = pd.concat((y_train, y_val), axis=0)
    sklearn_val_mask = np.hstack( (np.zeros(len(y_train),dtype=bool),  np.ones(len(y_val),dtype=bool)) )
    return X_train_val, y_train_val, sklearn_val_mask

rule sgdclassifier_optimize_optuna:
    output:
        (params_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        full_output_split_str+
        "/params_sgdclassifier_{n_trials}.json")
    input:
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}_dense"+pickle_ending
    run:
        (X_train, y_train, X_val, y_val)=load_dataset_splitted(input[0], input[1], wildcards=wildcards)
        X_train_val, y_train_val, sklearn_val_mask = create_joined_train_val_set_for_sgdclassifier(X_train, y_train, X_val, y_val)

        from graph_description.training_utils import sgdclassifier_objective

        objective = partial(sgdclassifier_objective,
                            sklearn_val_mask=sklearn_val_mask,
                            X_train_val=X_train_val,
                            y_train_val=y_train_val,
                            X_val=X_val,
                            y_val=y_val)

        run_and_save_study(
            objective=objective,
            study_name = f"{wildcards.dataset}-{wildcards.round}-{wildcards.num_train_per_class}-sgdclassifier",
            direction="maximize",
            n_trials=wildcards.n_trials,
            out_path=output[0]
        )
# snakemake "./snakemake_base/classifier_predictions/pubmed_planetoid/round_1/split_50_500_rest_0/sgdclassifier_opti(0, 220).npy" --cores=1



rule sgdclassifier_train_predict_opticlassifier:
    output :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        full_output_split_str+
        "/sgdclassifier_opti({opti_split_seed},{param_search_n_trials}).npy")
    input :
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_{round}_dense"+pickle_ending,
        (params_dir+
            "/{dataset}_{group}"+
            "/round_{round}"+
            "/split{split_mutator}{num_train_per_class}_{num_val}_{num_test}_{opti_split_seed}"+
            "/params_sgdclassifier_{param_search_n_trials}.json")
    run :
        (X_train, y_train, X_val, y_val, df)=load_dataset_splitted(input[0], input[1], return_val=True, return_full=True, wildcards=wildcards)
        X_train_val, y_train_val, sklearn_val_mask = create_joined_train_val_set_for_sgdclassifier(X_train, y_train, X_val, y_val)

        from graph_description.training_utils import sgdclassifier_train_classifier, TrialWrapperDict

        trial = TrialWrapperDict(input[2])
        clf = sgdclassifier_train_classifier(trial, sklearn_val_mask, X_train_val, y_train_val)

        prediction = clf.predict(df.drop("labels", axis=1))
        np.save(output[0], prediction, allow_pickle=False)
















def get_gnn_predictions_input_files(wildcards):
    #print(wildcards)
    if len(wildcards.optiparams)==0:
        return splits_dir+f"/{wildcards.dataset}_{wildcards.group}/{wildcards.num_train_per_class}_{wildcards.num_val}_{wildcards.num_test}_{wildcards.split_seed}"+npz_ending
    else:
        optiparams = wildcards.optiparams
        assert optiparams.startswith("_opti(")
        param_search_seed, param_search_n_trials = optiparams[len("_opti("):-1].split(",")
        #print(param_search_seed, param_search_n_trials)
        output= [
            splits_dir+f"/{wildcards.dataset}_{wildcards.group}/{wildcards.num_train_per_class}_{wildcards.num_val}_{wildcards.num_test}_{wildcards.split_seed}"+npz_ending,
             (params_dir+
                f"/{wildcards.dataset}_{wildcards.group}"+
                f"/baselines"+
                f"/split_{wildcards.num_train_per_class}_{wildcards.num_val}_{wildcards.num_test}_{param_search_seed}/params_{wildcards.gnn_kind}_{param_search_n_trials}.json")
        ]
        #print(output)
        return output



#print("WORKFLOW", )
gnn_ending = ".npy"
gnn_raw = "/{gnn_kind,[^_]+}_{init_seed,[^_]+}_{train_seed,[^_]+}{optiparams,.*}"
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
        get_gnn_predictions_input_files
    run :
        print("----------------------------------"  )
        from graph_description.gnn.run import main

        import numpy as np
        import pandas as pd
        splits = np.load(input[0])
        splits = {"train" : splits["train_mask"],
             "valid" : splits["val_mask"],
             "test" : splits["test_mask"]}

        if len(wildcards.optiparams)==0:

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

        else:
            from graph_description.training_utils import gnn_get_config, TrialWrapperDict
            from graph_description.gnn.run import main
            trial = TrialWrapperDict(input[1])
            cfg = gnn_get_config(trial, wildcards.gnn_kind, wildcards.dataset, wildcards.group)
            prediction = main(cfg, splits, init_seed=0, train_seed=0, silent=True)
        np.save(output[0], prediction, allow_pickle=False)

# snakemake "./snakemake_base/classifier_predictions/pubmed_planetoid/baselines/split_20_500_rest_1/gat2017_0_0.npy" --cores 1
# snakemake "./snakemake_base/classifier_predictions/pubmed_planetoid/baselines/split_20_500_rest_1/gat2017_0_0_opti(0 10).npy" --cores 1 -f

def process_score_name(score_name, path):
    from sklearn.metrics import accuracy_score, f1_score
    if "_train" in path:
        set_name = "train"
    elif "_val" in path:
        set_name = "val"
    elif "_test" in path:
        set_name = "test"
    else:
        set_name="test"
    assert set_name in ("test", "val", "train")
    scorers = {
            "accuracy" : accuracy_score,
            "f1" : partial(f1_score, average="micro")
    }
    return scorers[score_name], set_name

rule score_classifier:
    input :
        (prediction_dir+
        "/{dataset}_{group}"+
        "/{joker2}"+
        "/split{split_mutator}{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        "/{joker}.npy"),
        splits_dir+"/{dataset}_{group}/{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+npz_ending,
        aggregated_datasets_dir+"/{dataset}_{group}_0"+pickle_ending
    output :
        expand(scorer_dir+
        "/{{dataset}}_{{group}}"+
        "/{{joker2}}"+
        "/score_{{score_name}}{splits}"+
        "/split{{split_mutator,_|_labelsonly_}}{{num_train_per_class}}_{{num_val}}_{{num_test}}_{{split_seed}}"+
        "/{{joker}}"+txt_ending, splits=["", "_train", "_val"]),
    run :
        import numpy as np
        import pandas as pd
        for path in output:
            scorer, split_name = process_score_name(wildcards.score_name, path)
            splits = np.load(input[1])
            mask = splits[split_name+"_mask"]

            df  = pd.read_pickle(input[2])
            test_df = df[mask]
            y_test = test_df["labels"]

            y_test_predict = np.load(input[0])[mask]
            score = scorer(y_test, y_test_predict)
            score = np.array([score])
            np.savetxt(path, score)
# snakemake "./snakemake_base/scores/citeseer_planetoid/baselines/score_accuracy/split_20_500_rest_1/gat2017_0_0.txt" --cores 1




rule OLD_test_xgbclassifier:
    input :
        (classifier_dir+
        "/{dataset}_{group}"+
        "/round_{round}"+
        "/split_{num_train_per_class}_{num_val}_{num_test}_{split_seed}"+
        xgb_str_no_constr),
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

def c_range(the_stop, start=None):
    vals = [(1,6),
            (2,10),
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
    if start is None:
        start=1
    else:
        tmp=the_stop
        the_stop=start
        start=tmp
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


def unpack_wildcards(wildcards, _locals):
    for key in wildcards.keys():
        #print(key)
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
    #print("AAAA")
    unpack_wildcards(wildcards, globals())
    #print(globals())
    #print(n_estimators)
    result = [(scorer_dir+
        f"/{wildcards.dataset}_{wildcards.group}"+
        f"/{wildcards.round}"+
        f"/score_{wildcards.score_name}"+
        f"/split{wildcards.split_mutator}{num_train_per_class}_{wildcards.num_val}_{wildcards.num_test}_{split_seed}"+
        wildcards.joker+txt_ending)
        for num_train_per_class, split_seed in product(dyn_num_train_per_class, dyn_split_seed)]
    #print(result)
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
        "/split{split_mutator,_|_labelsonly_}{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
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



rule all_planetoid_xgb:
    input :
        expand((experiment_dir+
        "/agg_train_per_class"
        "/{dataset}_{group}"+
        "/{round}"+
        "/score_{{score_name}}"+
        "/split_{{dyn_num_train_per_class}}_{{num_val}}_{{num_test}}_{{dyn_split_seed}}"+
        "/{{joker}}"+csv_ending), dataset=["citeseer", "pubmed", "cora"], group=["planetoid"], round=["round_0","round_1","round_2"]),
    output :
        (experiment_dir+
        "/agg_train_per_class"
        "/all"+
        "/score_{score_name}"+
        "/split_{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
        "/{joker,[^g].*}"+csv_ending),
    shell: 'touch "{output}"'
#  snakemake "./snakemake_base/experiments/agg_train_per_class/all/score_accuracy/split_crange(5,201)_500_rest_range(20)/xgbclass_opti(0,100).csv" --cores 32 --wms-monitor "http://127.0.0.1:5000" --rerun-incomplete --retries 3
#  snakemake "./snakemake_base/experiments/agg_train_per_class/all/score_accuracy/split_crange(5,201)_500_rest_range(20)/rulefit10_opti(0,100).csv" --cores 32 --wms-monitor "http://127.0.0.1:5000" --rerun-incomplete --retries 3

datasets_per_group = {
    "planetoid" : ["citeseer", "pubmed", "cora"],
    "citationfull" :  ["citeseer", "pubmed", "cora_ml", "dblp", "cora"],
}
def remove_citationfull_cora2(l):
    out = []
    for s in l:
        if "citationfull" in s and "cora" in s and "cora_ml" not in s:
            continue
        if "_labelsonly_" in s and "round_0" in s:
            continue
        out.append(s)
    return out

for group, datasets in datasets_per_group.items():
    for mutator in ["_", "_labelsonly_"]:
        rule_name = f"make_all_{group}_{mutator}"
        rule:
            name :rule_name,
            input :
                remove_citationfull_cora2(expand((experiment_dir+
                "/agg_train_per_class"
                "/{dataset}_"+f"{group}"+
                "/{round}"+
                "/score_{{score_name}}{split}"+
                "/split{split_mutator}{{dyn_num_train_per_class}}_{{num_val}}_{{num_test}}_{{dyn_split_seed}}"+
                "/{{joker}}"+csv_ending),
                dataset=datasets, group=group,
                round=["round_0", "round_1", "round_2"],
                split=["", "_train", "_val"],
                split_mutator=[mutator])),
            output :
                (experiment_dir+
                "/agg_train_per_class"
                f"/{group}_all"+
                "/score_{score_name}"+
                f"/split{mutator}"+"{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
                "/{joker,[^g].*}"+csv_ending),
            shell: 'touch "{output}"'

#  snakemake "./snakemake_base/experiments/agg_train_per_class/all/score_accuracy/split_crange(5,201)_500_rest_range(20)/xgbclass_opti(0,100).csv" --cores 32 --wms-monitor "http://127.0.0.1:5000" --rerun-incomplete --retries 3
#  snakemake "./snakemake_base/experiments/agg_train_per_class/all/score_accuracy/split_crange(5,201)_500_rest_range(20)/rulefit10_opti(0,100).csv" --cores 32 --wms-monitor "http://127.0.0.1:5000" --rerun-incomplete --retries 3


rule all_one_gnn:
    input :
        expand((experiment_dir+
        "/agg_train_per_class"
        "/{dataset}_{group}"+
        "/baselines"+
        "/score_{{score_name}}"+
        "/split_{{dyn_num_train_per_class}}_{{num_val}}_{{num_test}}_{{dyn_split_seed}}"+
        "/{{joker}}"+csv_ending), dataset=["citeseer", "pubmed", "cora"], group=["planetoid"]),
    output :
        (experiment_dir+
        "/agg_train_per_class"
        "/all"+
        "/score_{score_name}"+
        "/split_{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
        "/{joker,(gat|gcn).*}"+csv_ending),
    shell: 'touch "{output}"'

rule all_both_gnn:
    input :
        expand((experiment_dir+
        "/agg_train_per_class"
        "/all"+
        "/score_{{score_name}}"+
        "/split_{{dyn_num_train_per_class}}_{{num_val}}_{{num_test}}_{{dyn_split_seed}}"+
        "/{gnn_kind}{{joker}}"+csv_ending), dataset=["citeseer", "pubmed", "cora"], group=["planetoid"], gnn_kind=["gat2017", "gcn2017"]),
    output :
        (experiment_dir+
        "/agg_train_per_class"
        "/all"+
        "/score_{score_name}"+
        "/split_{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
        "/gnns{joker,.*}"+csv_ending),
    shell: 'touch "{output}"'
#  snakemake "./snakemake_base/experiments/agg_train_per_class/all/score_accuracy/split_crange(5,201)_500_rest_range(20)/gat2017_0_0_opti(0,100).csv" --cores 4 --wms-monitor "http://127.0.0.1:5000" --rerun-incomplete --retries 3


for group, datasets in datasets_per_group.items():
    rule_name = f"gnn_make_all_{group}"
    rule:
        name : rule_name
        input :
            expand((experiment_dir+
            "/agg_train_per_class"
            "/{dataset}_"+f"{group}"+
            "/baselines"+
            "/score_{{score_name}}{split}"+
            "/split_{{dyn_num_train_per_class}}_{{num_val}}_{{num_test}}_{{dyn_split_seed}}"+
            "/{{joker}}"+csv_ending), dataset=datasets, group=group, split=["", "_train", "_val"]),
        output :
            (experiment_dir+
            "/agg_train_per_class"
            f"/{group}_all"+
            "/score_{score_name}"+
            "/split_{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
            "/{joker,(gat|gcn).*}"+csv_ending),
        shell: 'touch "{output}"'

    rule:
        name : f"gnn_both_make_all_{group}"
        input :
            expand((experiment_dir+
            "/agg_train_per_class"
            f"/{group}_all"+
            "/score_{{score_name}}"+
            "/split_{{dyn_num_train_per_class}}_{{num_val}}_{{num_test}}_{{dyn_split_seed}}"+
            "/{gnn_kind}{{joker}}"+csv_ending), dataset=datasets, group=group, gnn_kind=["gat2017", "gcn2017"]),
        output :
            (experiment_dir+
            "/agg_train_per_class"
            f"/{group}_all"+
            "/score_{score_name}"+
            "/split_{dyn_num_train_per_class,[^_]+}_{num_val,[0-9]+}_{num_test,rest|[0-9]+}_{dyn_split_seed,[^/\\\\]+}"+
            "/gnns{joker,.*}"+csv_ending),
        shell: 'touch "{output}"'




# snakemake .\snakemake_base\experiments\agg_train_per_class\citeseer_None\round_1\score_accuracy\split_auto_0_rest_10\xgbclass_10_2_0.csv --cores 1 -f

# snakemake .\snakemake_base\experiments\agg_train_per_class\citeseer_None\round_1\score_accuracy\split_auto_0_rest_10\xgbclass_10_2_0.csv --cores 1 -f



# snakemake ".\snakemake_base\experiments\agg_train_per_class\pubmed_planetoid\round_0\score_accuracy\split_1_0_rest_0\xgbclass_10_2_0.csv"

# snakemake ./snakemake_base/experiments/agg_train_per_class/pubmed_planetoid/round_0/score_accuracy/split_20_500_rest_0/xgbclass_10_2_0.csv --cores 1

#  snakemake "./snakemake_base/experiments/agg_train_per_class/pubmed_planetoid/round_0/score_accuracy/split_crange(200)_500_rest_0/xgbclass_10_2_0_5.csv" --cores 16


# snakemake "./snakemake_base/experiments/agg_train_per_class/pubmed_planetoid/baselines/score_accuracy/split_crange(200)_500_rest_0/gcn2017_0_0.csv" --cores 16 -f -R


# snakemake "./snakemake_base/experiments/agg_train_per_class/citeseer_planetoid/round_0/score_accuracy/split_crange(5,201)_500_rest_range(20)/xgbclass_opti(0,100).csv" --cores 16

