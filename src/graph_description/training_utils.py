
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


from xgboost import XGBClassifier
# 1. Define an objective function to be maximized.
def xgb_objective(trial, num_classes, dtrain, dval):
    init_lr = trial.suggest_float("init_lr", 0.001, 1)
    stop_lr = trial.suggest_float("init_lr", 0.001, init_lr)
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
        #max_delta_step=0.2
 #       reg_lambda = 1,
 #       reg_alpha = 0.5,
        colsample_bylevel = colsample,
        colsample_bynode = colsample,
        callbacks = (lr_scheduler, sample_scheduler, weight_scheduler),
#        booster =  trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        reg_lambda= trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        reg_alpha= trial.suggest_float('alpha', 1e-8, 1.0, log=True)
    )
    #bst = xgb.train(params, dtrain)
    bst = XGBClassifier( **params)

    result = bst.fit(dtrain.get_data(),dtrain.get_label(),
        eval_set=[(dval.get_data(),dval.get_label())],
        verbose=False)
    return bst.best_score