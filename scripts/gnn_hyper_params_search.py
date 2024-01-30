from pathlib import Path


prefix = Path(__file__).parent.parent



import argparse
parser = argparse.ArgumentParser(prog='optimize xgbs hyper parameters',
                    description='Uses fortuna to find optimal hyper parameters for xgboost',)
parser.add_argument("model")
parser.add_argument("--dataset")
parser.add_argument("--dataset_group", default="planetoid")
parser.add_argument("--train_per_class")
parser.add_argument("--seed", default=0)
parser.add_argument("--n_trials", default=100)

args = parser.parse_args()
print(args)
assert args.model in ("gat2017", "gcn2017")
gnn_kind = args.model
train_per_class = args.train_per_class
dataset_group = args.dataset_group

dataset=args.dataset


import os
import numpy as np
import optuna
from sklearn.metrics import accuracy_score

try:
    this_file = Path(__file__)
except NameError:
    this_file = Path(os.path.abspath(''))
if this_file.is_file():
    this_file=this_file.parent
if this_file.stem in ("notebooks", "scripts"):
    root_folder = this_file.parent
else:
    root_folder = this_file

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from graph_description.gnn.run import main


config_dir = root_folder/"src"/"graph_description"/"gnn"/"config"
data_root = root_folder/"pytorch_datasets"


input = [Path(prefix/f"snakemake_base/splits/{dataset}_{dataset_group}/{train_per_class}_500_rest_0.npz").resolve().absolute(),
         Path(prefix/f"snakemake_base/aggregated_datasets/{dataset}_{dataset_group}_{0}_labels.npy").resolve().absolute()]


splits = np.load(input[0])
splits = {"train" : splits["train_mask"],
     "valid" : splits["val_mask"],
     "test" : splits["test_mask"]}
train_mask = splits["train"]
val_mask = splits["valid"]

labels  = np.load(input[1])
print(labels)
print(val_mask)
y_val = labels[val_mask]


from graph_description.training_utils import gnn_objective

objective = partial(gnn_objective, gnn_kind=gnn_kind, dataset=dataset, splits=splits, y_val=y_val)

from functools import partial


storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(str(prefix/"hyper_param_journal.log")),
)

study = optuna.create_study(
    storage=storage,  # Specify the storage URL here.
    study_name=f"{dataset}-X-{train_per_class}-{gnn_kind}",
    load_if_exists=True,
    direction='maximize'
)

study.optimize(objective, n_trials=args.n_trials)