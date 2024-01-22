import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv


class GAT2017(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_output_heads: int,
        out_dim: int,
        n_layers: int = 2,
        dropout_p: float = 0.6,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {n_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GATConv(
                        in_dim, hidden_dim, heads=n_heads, dropout=dropout_p,
                    ),
                    "act": nn.ELU(),
                }
            )
        )
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": GATConv(
                            hidden_dim * n_heads,
                            hidden_dim,
                            heads=n_heads,
                            dropout=dropout_p,
                        ),
                        "act": nn.ELU(),
                    }
                )
            )
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GATConv(
                        hidden_dim * n_heads,
                        out_dim,
                        heads=n_output_heads,
                        dropout=dropout_p,
                        concat=False,
                    ),
                }
            )
        )

    def forward(self, data):
        x, edge_index = (data.x, data.adj_t)
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
        return x

    def activations(self, data):
        hs = {}
        x, edge_index = (data.x, data.adj_t)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x
        return hs


class GCN2017(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout_p: float = 0.6,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {n_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GCNConv(in_dim, hidden_dim),
                    "act": nn.ReLU(),
                }
            )
        )
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": GCNConv(hidden_dim, hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GCNConv(hidden_dim, out_dim),
                }
            )
        )

    def forward(self, data):
        x, edge_index = (data.x, data.adj_t)
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
        return x

    def activations(self, data):
        hs = {}
        x, edge_index = (data.x, data.adj_t)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x
        return hs



from typing import Any, Dict

import torch.nn
import torch_geometric.data
from omegaconf import DictConfig

from graph_description.gnn.models import GAT2017, GCN2017


def count_parameters(m: torch.nn.Module, trainable: bool = True) -> int:
    """Count the number of (trainable) parameters of a model

    Args:
        m (torch.nn.Module): model to count parameters of
        trainable (bool, optional): Whether to only count trainable parameters. Defaults to True.

    Returns:
        int: number of parameters
    """
    if trainable:
        return sum(w.numel() for w in m.parameters() if w.requires_grad)
    else:
        return sum(w.numel() for w in m.parameters())


def get_model(
    dataset: torch_geometric.data.Dataset, cfg: DictConfig
) -> torch.nn.Module:
    if cfg.name == "GAT2017":
        return get_GAT2017(dataset, cfg)
    elif cfg.name == "GCN2017":
        return get_GCN2017(dataset, cfg)
    else:
        raise ValueError(f"Unkown model name: {cfg.name}")


def get_GAT2017(
    dataset: torch_geometric.data.Dataset, cfg: DictConfig
) -> torch.nn.Module:
    assert isinstance(dataset.num_classes, int)
    return GAT2017(
        in_dim=dataset.num_features,
        out_dim=dataset.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout_p=cfg.dropout_p,
        n_heads=cfg.n_heads,
        n_output_heads=cfg.n_output_heads,
        n_layers=cfg.n_layers if hasattr(cfg, "n_layers") else 2,
    )


def get_GCN2017(
    dataset: torch_geometric.data.Dataset, cfg: DictConfig
) -> torch.nn.Module:
    assert isinstance(dataset.num_classes, int)
    return GCN2017(
        in_dim=dataset.num_features,
        out_dim=dataset.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout_p=cfg.dropout_p,
        n_layers=cfg.n_layers if hasattr(cfg, "n_layers") else 2,
    )