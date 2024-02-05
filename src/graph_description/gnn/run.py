import itertools
import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl

#import src.plots
#import src.similarity
#import src.similarity.predictions
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torch_geometric.data
import torch_geometric.transforms as T
import torch_geometric.utils
from omegaconf import DictConfig, OmegaConf
from graph_description.gnn.training import get_dataset, get_idx_split, train_node_classifier
from tqdm import tqdm

import pathlib


log = logging.getLogger(__name__)

config_path = pathlib.Path(__file__).parent.resolve()/"config"

def optional_hydrafunc(fn):
    if __name__ == "__main__":
        return hydra.main(config_path=str(config_path), config_name="main")(fn)
    else:
        return fn

@optional_hydrafunc
def main(cfg: DictConfig, splits=None, init_seed=0, train_seed=0, silent=False):
    import logging
    logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARNING)
    pl.seed_everything(cfg.datasplit_seed)
    # torch.use_deterministic_algorithms(
    #     True
    # )  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    if not silent:
        print(OmegaConf.to_yaml(cfg))


    dataset = get_dataset(
        name=cfg.dataset.name,
        group=cfg.dataset.group,
        root=cfg.data_root,
        transforms=[T.ToSparseTensor(remove_edge_index=False)],
        public_split=cfg.public_split,
        split_type="proportional",
        num_train_per_class=cfg.num_train_per_class,
        part_val=cfg.part_val,
        part_test=cfg.part_test,
    )
    if splits is None:
        splits = get_idx_split(dataset)
    import numpy as np
    for key, value in splits.items():
        if isinstance(value, np.ndarray):
            splits[key] = torch.from_numpy(value)
    # print(dataset.x, dataset.x.is_sparse)
    # print(dataset.y, dataset.y.is_sparse, dataset.y.shape)
    # print(dataset.edge_index, dataset.edge_index.is_sparse)
    # print(dataset.train_mask, dataset.train_mask.size())

    pl.seed_everything(cfg.seed)
    #for key, mask in splits.items():
    #    torch.save(mask, os.path.join(os.getcwd(), f"{key}_mask.pt"))
    splits["full"] = splits["train"] | splits["valid"] | splits["test"]

    model, data, eval_results = train_node_classifier(
                cfg, dataset, splits, init_seed=init_seed, train_seed=train_seed
    )
    import numpy as np
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_numpy = y_pred.cpu().detach().numpy()
    # print(y_numpy, np.bincount(y_numpy.ravel()))
    return y_numpy



if __name__ == "__main__":
    main()
