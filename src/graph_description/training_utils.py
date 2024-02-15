from threadpoolctl import threadpool_limits
import numpy as np
from sklearn.metrics import accuracy_score
def my_accuracy(y_true, y_pred):
    if len(y_pred.shape)>1:
        true_labels = np.argmax(y_pred, axis=1)
    else:
        true_labels=y_pred

    return -accuracy_score(y_true, true_labels)


from xgboost.callback import TrainingCallback, LearningRateScheduler

class LinearScheduler(TrainingCallback):
    def __init__(self, attr,  start_val, step, stop_val, timespan=1, silent=True, offset=0):
        self.attr = attr
        self.timespan = timespan
        self.start_val = start_val
        self.curr_val=start_val
        self.stop_val = stop_val
        self.step=step
        self.silent=silent
        self.offset=offset

        if step<0:
            self.agg = max
        else:
            self.agg = min

    def before_training(self, model):
        if not self.silent:
            print("new "+self.attr, self.curr_val)
        model.set_param(self.attr, self.curr_val)
        return model

    def after_iteration(self, model, epoch, evals_log):
        #print(model.attributes())
        if epoch>self.offset and epoch % (self.timespan) == 0:
            new_val = self.agg(self.stop_val, self.curr_val+self.step)
            assert new_val >= min(self.stop_val, self.start_val), str([new_val, self.step, self.start_val, self.step_val])
            assert new_val <= max(self.stop_val, self.start_val), str([new_val, self.step, self.start_val, self.step_val])
            if new_val!=self.curr_val:
                self.curr_val=new_val
                if not self.silent:
                    print("new "+self.attr, self.curr_val)
                model.set_param(self.attr, self.curr_val)


class ExponentialScheduler(TrainingCallback):
    def __init__(self, attr,  start_val, factor, stop_val, timespan=1, silent=True, offset=0):
        self.attr = attr
        self.timespan = timespan
        self.start_val = start_val
        self.curr_val=start_val
        self.stop_val = stop_val
        self.factor=factor
        self.silent=silent
        self.offset=offset

        if factor>1:
            self.agg = min
        else:
            self.agg = max

    def before_training(self, model):
        if not self.silent:
            print("new "+self.attr, self.curr_val)
        model.set_param(self.attr, self.curr_val)
        return model

    def after_iteration(self, model, epoch, evals_log):
        #print(model.attributes())
        if epoch>self.offset and epoch % (self.timespan) == 0:
            new_val = self.agg(self.stop_val, self.curr_val*self.factor)
            if new_val!=self.curr_val:
                self.curr_val=new_val
                if not self.silent:
                    print("new "+self.attr, self.curr_val)
                model.set_param(self.attr, self.curr_val)


class TrialWrapperDict:
    def __init__(self, d):
        if isinstance(d, dict):
            self.d = d
        elif isinstance(d,(Path, str)):
            import json
            with open(d, "r") as file:
                self.d=json.load(file)

    def suggest_float(self, name, *args, **kwargs):
        return self.d[name]

    def suggest_int(self, name, *args, **kwargs):
        return self.d[name]

    def suggest_categorical(self, name, *args, **kwargs):
        return self.d[name]

from xgboost import XGBClassifier

def xgb_get_config(trial, num_classes):
    init_lr = trial.suggest_float("init_lr", 0.001, 1)
    stop_lr = trial.suggest_float("stop_lr", 0.001, init_lr)
    lr_scheduler = ExponentialScheduler("learning_rate",
                                        start_val=init_lr,
                                        factor=trial.suggest_categorical("lr_factor",[0.95, 0.99, 1]),
                                        stop_val = stop_lr,
                                        timespan=1
                                       )
    start_subsample = trial.suggest_float("start_subsample", 0.1,1)
    stop_subsample = trial.suggest_float("stop_subsample", start_subsample,1)
    sample_scheduler = LinearScheduler("subsample",
                                        start_val=start_subsample,
                                        step=0.1,
                                        stop_val = stop_subsample,
                                        timespan=num_classes*trial.suggest_int("timespan_subsample",1,3),
                                        offset = num_classes*trial.suggest_int("timespan_offset_subsample",1,3)
                                       )

    start_weight = trial.suggest_int("start_weight", 1,10)
    stop_weight = trial.suggest_int("stop_weight", 1,10)
    step_weight = 1 if stop_subsample>=start_subsample else -1
    weight_scheduler = LinearScheduler("min_child_weight",
                                        start_val=start_weight,
                                        step=step_weight,
                                        stop_val = stop_weight,
                                        timespan=num_classes*trial.suggest_int("timespan_weight",1,3),
                                        offset = num_classes*trial.suggest_int("timespan_offset_weight",1,3)
                                       )
    #tree_size_scheduler = LinearScheduler("max_depth",1,1,10, timespan=10)
    colsample =  trial.suggest_float("colsample", 0.1,1)
    params = dict(
        n_estimators=500,
        max_depth=trial.suggest_int("max_depth", 1,15),
        learning_rate=1,
        objective='multi:softmax',
        random_state=0,
        eval_metric=my_accuracy,
        disable_default_eval_metric= 1,
        n_jobs=1,
        early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 1,100),
 #       min_child_weight=1,#child_scheduler.min_child_weight,
    #    multi_strategy='multi_output_tree',
    #    tree_method="approx",
    #    gamma=1,
 #       subsample=sample,
        colsample_bytree = colsample,
        colsample_bylevel = colsample,
        colsample_bynode = colsample,
        callbacks = (lr_scheduler, sample_scheduler, weight_scheduler),
#        booster =  trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        reg_lambda= trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        reg_alpha= trial.suggest_float('alpha', 1e-8, 1.0, log=True)
    )
    #bst = xgb.train(params, dtrain)
    return params


def xgb_objective(trial, num_classes, dtrain, dval):

    config = xgb_get_config(trial, num_classes)
    bst = XGBClassifier(**config)


    result = bst.fit(dtrain.get_data(),dtrain.get_label(),
        eval_set=[(dval.get_data(),dval.get_label())],
        verbose=False)
    return bst.best_score

from pathlib import Path
def get_root_folder():
    return Path(__file__).parent.parent.parent


def gnn_get_config(trial, gnn_kind, dataset, group):
    root_folder = get_root_folder()
    from hydra import compose, initialize_config_dir
    config_dir = root_folder/"src"/"graph_description"/"gnn"/"config"
    data_root = root_folder/"pytorch_datasets"
    with initialize_config_dir(config_dir=str(config_dir), job_name="test_app", version_base=None):
        cfg = compose(config_name="main",
                        overrides=["cuda=0",
                                    f"model={gnn_kind}",
                                    f"dataset={group}/{dataset}",
                                    f"data_root={data_root}",
                                    f"patience={trial.suggest_int('patience',0,100)}",
                                    f"optim.learning_rate={trial.suggest_float('lr',1e-3,100,log=True)}",
                                    f"optim.weight_decay={trial.suggest_float('lr_wdecay',0,.1)}",
                                    f"model.hidden_dim={trial.suggest_categorical('hidden_dim',[32,64,128,265])}",
                                    f"model.dropout_p={trial.suggest_float('dropout', 0,1)}",
                                    f"model.n_layers={trial.suggest_int('n_layers', 2,4)}",
        ])
        return cfg



def gnn_objective(trial, gnn_kind, dataset, group, splits, y_val):
    from graph_description.gnn.run import main
    cfg = gnn_get_config(trial, gnn_kind, dataset, group=group)
    prediction = main(cfg, splits, init_seed=0, train_seed=0, silent=True)
    val_prediction = prediction[splits["valid"]]
    return accuracy_score(val_prediction, y_val)


def rulefit_get_config(trial, max_rules):
    params = dict(
        max_rules=int(max_rules),
        cv=False,
        random_state=0,
        tree_size = trial.suggest_int('tree_size',2,100),
        memory_par = trial.suggest_float('memory_par',1e-3,100,log=True), # learning rate
        lin_trim_quantile = trial.suggest_float('lin_trim_quantile',0,1),
        exp_rand_tree_size = trial.suggest_categorical('exp_rand_tree_size',[False, True]),
#        alpha = trial.suggest_float('alpha',1e-4, 10, log=True),
    )
    return params


def rulefit_train_classifier(trial, max_rules, X_train, y_train):
    from imodels import RuleFitClassifier
    from sklearn.multiclass import OneVsRestClassifier
    import warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar subtract")
    warnings.filterwarnings("ignore", message="overflow encountered in reduce")

    params = rulefit_get_config(trial, max_rules)
    with threadpool_limits(limits=1, user_api='blas'):
        with threadpool_limits(limits=1, user_api='openmp'):
            clf = OneVsRestClassifier(RuleFitClassifier(**params))
            clf.fit(X_train, y_train)
    return clf


def rulefit_objective(trial, max_rules, X_train, y_train, X_val, y_val):
    clf = rulefit_train_classifier(trial, max_rules, X_train, y_train)
    prediction = clf.predict(X_val)
    return accuracy_score(prediction, y_val)







from graph_description.custom_sgdclassifier import SGDClassifierFixedSplit


def sgdclassifier_get_config(trial, sklearn_val_mask):
    loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge",
                                                "perceptron", "squared_error", "huber", "epsilon_insensitive",
                                                "squared_epsilon_insensitive"])
    if loss in ("huber", "epsilon_insensitive","squared_epsilon_insensitive"):
        epsilon=trial.suggest_float("epsilon", 0.01, 10, log=True)
    else:
        epsilon=0.1

    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    else:
        l1_ratio=0.15
    params = dict(
        loss=loss,
        penalty=penalty,
        alpha=trial.suggest_float("alpha", 1e-6, 10, log=True),
        l1_ratio=l1_ratio,
        tol = trial.suggest_float("tol", 1e-6, 0.1, log=True),
        epsilon=epsilon,
        random_state=0,
        n_jobs=1,

        early_stopping=True,
        validation_mask=sklearn_val_mask,
        n_iter_no_change=trial.suggest_int('n_iter_no_change',1,100),
        verbose=False,
    )
    return params


def sgdclassifier_train_classifier(trial, sklearn_val_mask, X_train_val, y_train_val):
    params = sgdclassifier_get_config(trial, sklearn_val_mask)

    # we need to assign threadpool limits, otherwise the calculation will be run in parallel
    # parallelism is not desired here but handled by snakemake instead
    with threadpool_limits(limits=1, user_api='blas'):
        with threadpool_limits(limits=1, user_api='openmp'):
            clf = SGDClassifierFixedSplit(**params)
            clf.fit(X_train_val, y_train_val)
    return clf


def sgdclassifier_objective(trial, sklearn_val_mask, X_train_val, y_train_val, X_val, y_val):
    clf = sgdclassifier_train_classifier(trial, sklearn_val_mask, X_train_val, y_train_val)
    prediction = clf.predict(X_val)
    score = accuracy_score(prediction, y_val)
    #print("final_score", score)
    return score