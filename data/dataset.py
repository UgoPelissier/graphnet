import os
import os.path as osp
import glob
import torch
import numpy as np
from typing import Optional, Callable
from torch_geometric.data import Dataset, Data
import numpy as np
import enum
import meshio
from alive_progress import alive_bar
from utils.utils import cell2point


class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    OBSTACLE = 1
    INFLOW = 2
    OUTFLOW = 3
    WALL_BOUNDARY = 4
    SIZE = 5


class MeshDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            dataset_name: str,
            u_0: float,
            split: str,
            indices: np.ndarray,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None
    ) -> None:
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.u_0 = u_0
        self.split = split
        self.idx = indices

        self.eps = torch.tensor(1e-8)

        # mean and std of the node features are calculated
        self.mean_vec_x = torch.zeros(11)
        self.std_vec_x = torch.zeros(11)

        # mean and std of the edge features are calculated
        self.mean_vec_edge = torch.zeros(3)
        self.std_vec_edge = torch.zeros(3)

        # mean and std of the output parameters are calculated
        self.mean_vec_y = torch.zeros(2)
        self.std_vec_y = torch.zeros(2)

        # define counters used in normalization
        self.num_accs_x  =  0
        self.num_accs_edge = 0
        self.num_accs_y = 0

        super().__init__(osp.join(self.data_dir, self.dataset_name), transform, pre_transform)

    @property
    def raw_file_names(self) -> list: 
        return ["stokes_{:03d}.vtu".format(i) for i in self.idx]

    @property
    def processed_file_names(self) -> list:
        return glob.glob(os.path.join(self.processed_dir, self.split, 'data_*.pt'))
    
    def download(self) -> None:
        pass

    def triangles_to_edges(self, faces: torch.Tensor) -> torch.Tensor:
        """Computes mesh edges from triangles."""
        # collect edges from triangles
        edges = torch.vstack((faces[:, 0:2],
                              faces[:, 1:3],
                              torch.hstack((faces[:, 2].unsqueeze(dim=-1),
                                            faces[:, 0].unsqueeze(dim=-1)))
                            ))
        receivers = torch.min(edges, dim=1).values
        senders = torch.max(edges, dim=1).values
        packed_edges = torch.stack([senders, receivers], dim=1)
        # remove duplicates and unpack
        unique_edges = torch.unique(packed_edges, dim=0)
        senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
        # create two-way connectivity
        return torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

    def update_stats(self, x: torch.Tensor, edge_attr: torch.Tensor, y: torch.Tensor) -> None:
        """Update the mean and std of the node features, edge features, and output parameters."""
        self.mean_vec_x += torch.sum(x, dim = 0)
        self.std_vec_x += torch.sum(x**2, dim = 0)
        self.num_accs_x += x.shape[0]

        self.mean_vec_edge += torch.sum(edge_attr, dim=0)
        self.std_vec_edge += torch.sum(edge_attr**2, dim=0)
        self.num_accs_edge += edge_attr.shape[0]

        self.mean_vec_y += torch.sum(y, dim=0)
        self.std_vec_y += torch.sum(y**2, dim=0)
        self.num_accs_y += y.shape[0]

    def save_stats(self) -> None:
        """Save the mean and std of the node features, edge features, and output parameters."""
        self.mean_vec_x = self.mean_vec_x / self.num_accs_x
        self.std_vec_x = torch.maximum(torch.sqrt(self.std_vec_x / self.num_accs_x - self.mean_vec_x**2), self.eps)

        self.mean_vec_edge = self.mean_vec_edge / self.num_accs_edge
        self.std_vec_edge = torch.maximum(torch.sqrt(self.std_vec_edge / self.num_accs_edge - self.mean_vec_edge**2), self.eps)

        self.mean_vec_y = self.mean_vec_y / self.num_accs_y
        self.std_vec_y = torch.maximum(torch.sqrt(self.std_vec_y / self.num_accs_y - self.mean_vec_y**2), self.eps)

        save_dir = osp.join(self.processed_dir, 'stats', self.split)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.mean_vec_x, osp.join(save_dir, 'mean_vec_x.pt'))
        torch.save(self.std_vec_x, osp.join(save_dir, 'std_vec_x.pt'))

        torch.save(self.mean_vec_edge, osp.join(save_dir, 'mean_vec_edge.pt'))
        torch.save(self.std_vec_edge, osp.join(save_dir, 'std_vec_edge.pt'))

        torch.save(self.mean_vec_y, osp.join(save_dir, 'mean_vec_y.pt'))
        torch.save(self.std_vec_y, osp.join(save_dir, 'std_vec_y.pt'))

    def process(self) -> None:
        """Process the dataset."""
        os.makedirs(os.path.join(self.processed_dir, self.split), exist_ok=True)

        data_list = []
        print(f'{self.split} dataset')
        with alive_bar(total=len(self.processed_file_names)) as bar:
            for idx, data in enumerate(self.raw_file_names):
                mesh = meshio.read(osp.join(self.raw_dir, data))

                # velocity field
                v = torch.Tensor(np.stack((cell2point(osp.join(self.raw_dir, data), 'u'), cell2point(osp.join(self.raw_dir, data), 'v'))).transpose())

                # node type
                node_type = torch.zeros(mesh.points.shape[0])
                for i in range(mesh.cells[1].data.shape[0]):
                    for j in range(mesh.cells[1].data.shape[1]):
                        node_type[mesh.cells[1].data[i,j]] = mesh.cell_data['Label'][1][i]

                # get edge indices in COO format
                edge_index = self.triangles_to_edges(torch.Tensor(mesh.cells[0].data)).long()

                # get edge attributes
                u_i = mesh.points[edge_index[0]]
                u_j = mesh.points[edge_index[1]]
                u_ij = torch.Tensor(u_i - u_j)
                u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
                edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

                # node outputs, for training (velocity)
                y = v.type(torch.float)

    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int) -> Data:
        data = torch.load(os.path.join(self.processed_dir, self.split, f'data_{idx}.pt'))
        return data
