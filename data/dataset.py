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
from graphnet.utils.utils import cell2point


class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    INFLOW = 1
    OUTFLOW = 2
    WALL_BOUNDARY = 3
    OBSTACLE = 4
    SIZE = 5


class MeshDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            dim: int,
            u_0: float,
            v_0: float,
            w_0: float,
            split: str,
            indices: np.ndarray,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None
    ) -> None:
        self.data_dir = data_dir
        self.dim = dim
        self.u_0 = u_0
        self.v_0 = v_0
        self.w_0 = w_0
        self.split = split
        self.idx = indices

        self.eps = torch.tensor(1e-8)

        # mean and std of the node features are calculated
        self.vec_x_size = self.dim + NodeType.SIZE
        self.mean_vec_x = torch.zeros(self.vec_x_size)
        self.std_vec_x = torch.zeros(self.vec_x_size)

        # mean and std of the edge features are calculated
        self.vec_edge_size = self.dim + 1
        self.mean_vec_edge = torch.zeros(self.vec_edge_size)
        self.std_vec_edge = torch.zeros(self.vec_edge_size)

        # mean and std of the output parameters are calculated
        self.mean_vec_y = torch.zeros(self.dim)
        self.std_vec_y = torch.zeros(self.dim)

        # define counters used in normalization
        self.num_accs_x  =  0
        self.num_accs_edge = 0
        self.num_accs_y = 0

        super().__init__(self.data_dir, transform, pre_transform)

    @property
    def raw_file_names(self) -> list:
        if self.dim==2:
            return ["stokes_{:03d}.vtu".format(i) for i in self.idx]
        elif self.dim==3:
            return ["cad_{:03d}.vtu".format(i) for i in self.idx]
        else:
            raise ValueError("The dimension must be either 2 or 3.")

    @property
    def processed_file_names(self) -> list:
        if self.dim==2:
            return glob.glob(os.path.join(self.processed_dir, self.split, 'stokes_*.pt'))
        elif self.dim==3:
            return glob.glob(os.path.join(self.processed_dir, self.split, 'cad_*.pt'))
        else:
            raise ValueError("The dimension must be either 2 or 3.")
        
    
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

    def tetra_to_edges(self, faces: torch.Tensor) -> torch.Tensor:
        """Computes mesh edges from triangles."""
        # collect edges from tertahedra
        edges = torch.vstack((faces[:, 0:2],
                            faces[:, 1:3],
                            faces[:, 2:4],
                            torch.hstack((faces[:, 3].unsqueeze(dim=-1),
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

        print(f'{self.split} dataset')
        with alive_bar(total=len(self.processed_file_names)) as bar:
            for data in self.raw_file_names:
                # read vtu file
                mesh = meshio.read(osp.join(self.raw_dir, 'sol', data))

                # node type
                node_type = torch.zeros(mesh.points.shape[0])
                for i in range(mesh.cells[1].data.shape[0]):
                    for j in range(mesh.cells[1].data.shape[1]):
                        if (self.dim == 2):
                            node_type[mesh.cells[1].data[i,j]] = mesh.cell_data['Label'][1][i]
                        elif (self.dim==3):
                            if (mesh.cell_data['Label'][1][i] == 31) or (mesh.cell_data['Label'][1][i] == 32):
                                node_type[mesh.cells[1].data[i,j]] = 3
                            else:
                                node_type[mesh.cells[1].data[i,j]] = mesh.cell_data['Label'][1][i]
                        else:
                            raise ValueError("The dimension must be either 2 or 3.")

                # get initial velocity
                v_0 = torch.zeros(mesh.points.shape[0], self.dim)
                mask = (node_type.long())==torch.tensor(NodeType.INFLOW)
                if (self.dim == 2):
                    v_0[mask] = torch.Tensor([self.u_0, self.v_0])
                elif (self.dim == 3):
                    v_0[mask] = torch.Tensor([self.u_0, self.v_0, self.w_0])
                else:
                    raise ValueError("The dimension must be either 2 or 3.")

                node_type_one_hot = torch.nn.functional.one_hot(node_type.long(), num_classes=NodeType.SIZE)

                # get features
                x = torch.cat((v_0, node_type_one_hot),dim=-1).type(torch.float)

                # get edge indices in COO format
                if (self.dim == 2):
                    edge_index = self.triangles_to_edges(torch.Tensor(mesh.cells[0].data)).long()
                elif (self.dim == 3):
                    edge_index = self.tetra_to_edges(torch.Tensor(mesh.cells[0].data)).long()
                else:
                    raise ValueError("The dimension must be either 2 or 3.")

                # get edge attributes
                u_i = mesh.points[edge_index[0]][:,:self.dim]
                u_j = mesh.points[edge_index[1]][:,:self.dim]
                u_ij = torch.Tensor(u_i - u_j)
                u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
                edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

                # node outputs, for training (velocity)
                if self.dim == 2:
                    v = torch.Tensor(np.stack((cell2point(osp.join(self.raw_dir, 'sol', data), 'u'), cell2point(osp.join(self.raw_dir, 'sol', data), 'v'))).transpose())
                elif self.dim == 3:
                    v = torch.Tensor(np.stack((cell2point(osp.join(self.raw_dir, 'sol', data), 'u'), cell2point(osp.join(self.raw_dir, 'sol', data), 'v'), cell2point(osp.join(self.raw_dir, 'sol', data), 'w'))).transpose())
                else:
                    raise ValueError("The dimension must be either 2 or 3.")
                y = v.type(torch.float)

                self.update_stats(x, edge_attr, y)

                torch.save(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cells=torch.Tensor(mesh.cells[0].data), mesh_pos=torch.Tensor(mesh.points), n_points=x.shape[0], n_edges=edge_index.shape[1], n_cells=mesh.cells[0].data.shape[0], v_0=v_0, name=data[:-4]),
                            osp.join(self.processed_dir, self.split, f'{data[:-4]}.pt'))
                
                bar()
                    
        self.save_stats()

    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int) -> Data:
        if self.dim==2:
            data = torch.load(os.path.join(self.processed_dir, self.split, "stokes_{:03d}.pt".format(self.idx[idx])))
        elif self.dim==3:
            data = torch.load(os.path.join(self.processed_dir, self.split, "cad_{:03d}.pt".format(self.idx[idx])))
        else:
            raise ValueError("The dimension must be either 2 or 3.")
        return data
