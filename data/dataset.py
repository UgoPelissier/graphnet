import os
import os.path as osp
import torch
from typing import Optional, Callable
from torch_geometric.data import Dataset, Data, download_url
import numpy as np
import json
import tensorflow as tf
import functools
import enum
from utils.utils import progressBar
from alive_progress import alive_bar


class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


class MeshDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 dataset_name: str,
                 field: str,
                 time_steps: int,
                 idx_lim_train: int,
                 idx_lim_val: int,
                 idx_lim_test: int,
                 time_step_lim: int,
                 split: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None
    ) -> None:
        self.split = split
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.field = field
        self.time_steps = time_steps

        if self.split == 'train':
            self.idx_lim = idx_lim_train
        elif self.split == 'valid':
            self.idx_lim = idx_lim_val
        elif self.split == 'test':
            self.idx_lim = idx_lim_test
        else:
            raise ValueError(f"Invalid split: {self.split}")
        self.time_step_lim = time_step_lim

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

        super().__init__(self.data_dir, transform, pre_transform)

    @property
    def raw_file_names(self) -> list: 
        return ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']

    @property
    def processed_file_names(self) -> list:
        return ["train.pt", "valid.pt", "test.pt"]
    
    def download(self) -> None:
        print(f'Download dataset {self.dataset_name} to {self.raw_dir}')
        for file in ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']:
            url = f"https://storage.googleapis.com/dm-meshgraphnets/{self.dataset_name}/{file}"
            download_url(url=url, folder=self.raw_dir)

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

    def _parse(self, proto, meta: dict) -> dict:
        """Parses a trajectory from tf.Example."""
        feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta['features'].items():
            data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
            data = tf.reshape(data, field['shape'])
            out[key] = data
        return out

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

        save_dir = osp.join(self.processed_dir, 'stats', self.split, )
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.mean_vec_x, osp.join(save_dir, 'mean_vec_x.pt'))
        torch.save(self.std_vec_x, osp.join(save_dir, 'std_vec_x.pt'))

        torch.save(self.mean_vec_edge, osp.join(save_dir, 'mean_vec_edge.pt'))
        torch.save(self.std_vec_edge, osp.join(save_dir, 'std_vec_edge.pt'))

        torch.save(self.mean_vec_y, osp.join(save_dir, 'mean_vec_y.pt'))
        torch.save(self.std_vec_y, osp.join(save_dir, 'std_vec_y.pt'))

    def process(self) -> None:
        """Process the dataset."""
        # load meta data
        with open(osp.join(self.raw_dir, 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        self.dt = meta['dt']
        # convert data to dict
        ds = tf.data.TFRecordDataset(osp.join(self.raw_dir, f'%s.tfrecord' % self.split))
        ds = ds.map(functools.partial(self._parse, meta=meta), num_parallel_calls=8)

        data_list = []
        print(f'{self.split} dataset')
        with alive_bar(total=self.idx_lim) as bar:
            for idx, data in enumerate(ds):
                if (idx==self.idx_lim):
                    break
                bar()
                # convert tensors from tf to pytorch
                d = {}
                for key, value in data.items():
                        d[key] = torch.from_numpy(value.numpy()).squeeze(dim=0)
                # extract data from each time step
                for t in range(self.time_steps-1):
                    if (t==self.time_step_lim):
                        break
                    # get node features
                    v = d['velocity'][t, :, :]
                    node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data['node_type'][0,:,0]), NodeType.SIZE)))
                    x = torch.cat((v, node_type),dim=-1).type(torch.float)

                    # get edge indices in COO format
                    edge_index = self.triangles_to_edges(d['cells']).long()

                    # get edge attributes
                    u_i = d['mesh_pos'][edge_index[0]]
                    u_j = d['mesh_pos'][edge_index[1]]
                    u_ij = u_i - u_j
                    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
                    edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

                    # node outputs, for training (velocity)
                    v_t = d['velocity'][t, :, :]
                    v_tp1 = d['velocity'][t+1, :, :]
                    y = ((v_tp1-v_t)/meta['dt']).type(torch.float)

                    self.update_stats(x, edge_attr, y)
                    data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cells=d['cells'], mesh_pos=d['mesh_pos'], n_points=x.shape[0], n_edges=edge_index.shape[1], n_cells=d['cells'].shape[0]))

        torch.save(data_list, osp.join(self.processed_dir, f'{self.split}.pt'))
        self.save_stats()

    def len(self) -> int:
        return self.idx_lim*(self.time_step_lim-1)
    
    def get(self, idx: int) -> Data:
        data = torch.load(osp.join(self.processed_dir, f'{self.split}.pt'))
        return data[idx]
