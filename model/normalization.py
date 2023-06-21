import torch
from typing import List
from torch_geometric.data import Data

def normalize(to_normalize: torch.Tensor, mean_vec: float, std_vec: float) -> torch.Tensor:
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize: torch.Tensor, mean_vec: float, std_vec: float) -> torch.Tensor:
    return to_unnormalize*std_vec+mean_vec