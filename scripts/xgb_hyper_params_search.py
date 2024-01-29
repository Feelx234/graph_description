import optuna
from pathlib import Path
import numpy as np
import pandas as pd

prefix = Path(__file__).parent.parent



import argparse
parser = argparse.ArgumentParser(prog='optimize xgbs hyper parameters',
                    description='Uses fortuna to find optimal hyper parameters for xgboost',)

parser.add_argument("--dataset")
parser.add_argument("--round")
parser.add_argument("--train_per_class")
parser.add_argument("--seed", default=0)

args = parser.parse_args()
print(args)

train_per_class = args.train_per_class
round = args.round
dataset=args.dataset

#exit()



input = [Path(prefix/f"snakemake_base/splits/{dataset}_planetoid/{train_per_class}_500_rest_0.npz").resolve().absolute(),
         Path(prefix/f"snakemake_base/aggregated_datasets/{dataset}_planetoid_{round}.pkl").resolve().absolute()]


splits = np.load(input[0])
train_mask = splits["train_mask"]
val_mask = splits["val_mask"]

df  = pd.read_pickle(input[1])

train_df = df[train_mask]
X_train = train_df.drop("labels", axis=1)
y_train = train_df["labels"]

val_df = df[val_mask]
X_val = val_df.drop("labels", axis=1)
y_val = val_df["labels"]

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
dval = xgb.DMatrix(X_val, label=y_val, missing=np.NaN)

num_classes = len(np.bincount(y_train))

from graph_description.training_utils import xgb_objective
from functools import partial

objective = partial(xgb_objective, num_classes=num_classes, dtrain=dtrain, dval=dval)


storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(str(prefix/"hyper_param_journal.log")),
)

study = optuna.create_study(
    storage=storage,  # Specify the storage URL here.
    study_name=f"{dataset}-{round}-{train_per_class}",
    load_if_exists=True,
    direction='minimize'
)

study.optimize(objective, n_trials=100)