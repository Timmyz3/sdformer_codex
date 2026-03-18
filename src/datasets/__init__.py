"""Dataset builders."""

from .optical_flow_dsec import DSECFlowDataset
from .optical_flow_mvsec import MVSECFlowDataset


def build_dataset(cfg, split: str):
    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "dsec":
        return DSECFlowDataset(cfg, split)
    if dataset_name == "mvsec":
        return MVSECFlowDataset(cfg, split)
    raise KeyError(f"unsupported dataset: {dataset_name}")

