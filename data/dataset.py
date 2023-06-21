import os
import os.path as osp
import torch
from typing import Optional, Callable
from torch_geometric.data import Dataset, Data, download_url
import numpy as np
import glob
import json
import tensorflow as tf
import functools
import enum


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
                 history: bool,
                 split: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None
    ) -> None:
        self.split = split
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.field = field
        self.history = history

        self.idx_lim = 10 if self.split == 'train' else 2

        super().__init__(self.data_dir, transform, pre_transform)

    @property
    def raw_file_names(self) -> list: 
        return ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']

    @property
    def processed_file_names(self) -> list:
        return glob.glob(os.path.join(self.processed_dir, self.split, 'data_*.pt'))
    
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

    def process(self) -> None:
        with open(os.path.join(self.raw_dir, 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        ds = tf.data.TFRecordDataset(os.path.join(self.raw_dir, f'%s.tfrecord' % self.split))
        ds = ds.map(functools.partial(self._parse, meta=meta), num_parallel_calls=8)

        os.makedirs(os.path.join(self.processed_dir, self.split), exist_ok=True)
        for idx, data in enumerate(ds):
            data_list = []
            if (idx==self.idx_lim):
                break
            d = {}
            for key, value in data.items():
                    d[key] = torch.from_numpy(value.numpy()).squeeze(dim=0)
            ts = d['velocity'].shape[0]
            for t in range(ts-1):
                # get node features
                v = d['velocity'][t, :, :]
                node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data['node_type'][0,:,0]), NodeType.SIZE)))
                x = torch.cat((v, node_type),dim=-1).type(torch.float)

                # get edge indices in COO format
                edge_index = self.triangles_to_edges(d['cells'])

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

                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, p=d['pressure'][t, :, :], cells=d['cells'], mesh_pos=d['mesh_pos']))

            torch.save(data_list, os.path.join(self.processed_dir, self.split, f'data_{idx}.pt'))

    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int) -> Data:
        data = torch.load(os.path.join(self.processed_dir, self.split, f'data_{idx}.pt'))
        return data
