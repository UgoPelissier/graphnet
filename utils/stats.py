import os.path as osp
from typing import List, Tuple, Union
import torch


def load_stats(
        data_dir: str,
        device: torch.device
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load statistics from the dataset."""
    train_dir = osp.join(data_dir, 'processed', 'stats', 'train')
    mean_vec_x_train = torch.load(osp.join(train_dir, 'mean_vec_x.pt'), map_location=device)
    std_vec_x_train = torch.load(osp.join(train_dir, 'std_vec_x.pt'), map_location=device)
    mean_vec_edge_train = torch.load(osp.join(train_dir, 'mean_vec_edge.pt'), map_location=device)
    std_vec_edge_train = torch.load(osp.join(train_dir, 'std_vec_edge.pt'), map_location=device)
    mean_vec_y_train = torch.load(osp.join(train_dir, 'mean_vec_y.pt'), map_location=device)
    std_vec_y_train = torch.load(osp.join(train_dir, 'std_vec_y.pt'), map_location=device)
    train_stats = (mean_vec_x_train, std_vec_x_train, mean_vec_edge_train, std_vec_edge_train, mean_vec_y_train, std_vec_y_train)

    val_dir = osp.join(data_dir, 'processed', 'stats', 'valid')
    mean_vec_x_val = torch.load(osp.join(val_dir, 'mean_vec_x.pt'), map_location=device)
    std_vec_x_val = torch.load(osp.join(val_dir, 'std_vec_x.pt'), map_location=device)
    mean_vec_edge_val = torch.load(osp.join(val_dir, 'mean_vec_edge.pt'), map_location=device)
    std_vec_edge_val = torch.load(osp.join(val_dir, 'std_vec_edge.pt'), map_location=device)
    mean_vec_y_val = torch.load(osp.join(val_dir, 'mean_vec_y.pt'), map_location=device)
    std_vec_y_val = torch.load(osp.join(val_dir, 'std_vec_y.pt'), map_location=device)
    val_stats = (mean_vec_x_val, std_vec_x_val, mean_vec_edge_val, std_vec_edge_val, mean_vec_y_val, std_vec_y_val)

    test_dir = osp.join(data_dir, 'processed', 'stats', 'test')
    mean_vec_x_test = torch.load(osp.join(test_dir, 'mean_vec_x.pt'), map_location=device)
    std_vec_x_test = torch.load(osp.join(test_dir, 'std_vec_x.pt'), map_location=device)
    mean_vec_edge_test = torch.load(osp.join(test_dir, 'mean_vec_edge.pt'), map_location=device)
    std_vec_edge_test = torch.load(osp.join(test_dir, 'std_vec_edge.pt'), map_location=device)
    mean_vec_y_test = torch.load(osp.join(test_dir, 'mean_vec_y.pt'), map_location=device)
    std_vec_y_test = torch.load(osp.join(test_dir, 'std_vec_y.pt'), map_location=device)
    test_stats = (mean_vec_x_test, std_vec_x_test, mean_vec_edge_test, std_vec_edge_test, mean_vec_y_test, std_vec_y_test)

    return train_stats, val_stats, test_stats


def normalize(
        data: Union[torch.Tensor, List[torch.Tensor]],
        mean: Union[torch.Tensor, List[torch.Tensor]],
        std: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Normalize the data."""
    if isinstance(data, list):
        return [normalize(d, m, s) for d, m, s in zip(data, mean, std)] # type: ignore
    return (data - mean) / std


def unnormalize(
        data: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> torch.Tensor:
    """Normalize the data."""
    if isinstance(data, list):
        return [unnormalize(d, m, s) for d, m, s in zip(data, mean, std)] # type: ignore
    return (data * std) + mean