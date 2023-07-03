import copy
from typing import Optional, List, Tuple, Union
import os.path as osp
import json
import numpy as np

from utils.stats import normalize, unnormalize, load_stats
from utils.utils import get_next_version
from utils.vizu import save_vtu
from data.dataset import NodeType
from model.processor import ProcessorLayer

import torch
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER

from torch_geometric.data import Data

import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


class MeshGraphNet(pl.LightningModule):
    """Lightning module for the MeshNet model."""
    def __init__(
            self,
            path: str,
            dataset: str,
            logs: str,
            noise_std: float,
            num_layers: int,
            input_dim_node: int,
            input_dim_edge: int,
            hidden_dim: int,
            output_dim: int,
            optimizer: OptimizerCallable,
            test_indices: List[int],
            animate: bool,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()

        self.path = path
        self.dataset = dataset
        self.logs = logs
        self.noise_std = noise_std
        self.num_layers = num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                    #    ReLU(),
                                    #    Linear(hidden_dim, hidden_dim),
                                    #    ReLU(),
                                    #    Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                    #    ReLU(),
                                    #    Linear(hidden_dim, hidden_dim),
                                    #    ReLU(),
                                    #    Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))


        self.processor = torch.nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                #   Linear(hidden_dim, hidden_dim),
                                #   ReLU(),
                                #   Linear(hidden_dim, hidden_dim),
                                #   ReLU(),
                                  Linear(hidden_dim, output_dim))

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.test_indices = test_indices
        self.animate = animate
        self.version = f'version_{get_next_version(self.logs)}'

        with open(osp.join(self.dataset, 'raw', 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        self.dt = meta['dt']
        
    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, batch: Data, split: str):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr
        # x += self.v_noise(batch, self.noise_std)

        if split == 'train':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_train, self.mean_vec_edge_train], std=[self.std_vec_x_train, self.std_vec_edge_train])
        elif split == 'val':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_val, self.mean_vec_edge_val], std=[self.std_vec_x_val, self.std_vec_edge_val])
        elif split == 'test':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_test, self.mean_vec_edge_test], std=[self.std_vec_x_test, self.std_vec_edge_test])
        else:
            raise ValueError(f'Invalid split: {split}')

        # step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def loss(self, pred: torch.Tensor, inputs: Data, split: str) -> torch.Tensor:
        """Calculate the loss for the given prediction and inputs."""
        # get the loss mask for the nodes of the types we calculate loss for
        loss_mask=torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(NodeType.NORMAL)),
                                   (torch.argmax(inputs.x[:,2:],dim=1)==torch.tensor(NodeType.OUTFLOW)))

        # normalize labels with dataset statistics
        if split == 'train':
            labels = normalize(data=inputs.y, mean=self.mean_vec_y_train, std=self.std_vec_y_train)
        elif split == 'val':
            labels = normalize(data=inputs.y, mean=self.mean_vec_y_val, std=self.std_vec_y_val)
        elif split == 'test':
            labels = normalize(data=inputs.y, mean=self.mean_vec_y_test, std=self.std_vec_y_test)
        else:
            raise ValueError(f'Invalid split: {split}')

        # find sum of square errors
        error = torch.sum((labels-pred)**2, dim=1)

        # root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        pred = self(batch, split='train')
        loss = self.loss(pred, batch, split='train')
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        if self.trainer.sanity_checking:
            self.load_stats()
        pred = self(batch, split='val')
        loss = self.loss(pred, batch, split='val')
        self.log('valid/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> None:
        """Test step of the model."""
        self.load_stats()

        if batch_idx in self.test_indices:
            pred = self(batch, split='test')
            loss = self.loss(pred, batch, split='test')
            self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            data_list_true, data_list_prediction, data_list_error = self.rollout(batch, batch_idx)

            if self.animate:
                save_vtu(data_list_true, data_list_prediction, data_list_error, path=osp.join(self.logs, self.version))

    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], List[Union[_TORCH_LRSCHEDULER, ReduceLROnPlateau]]]]:
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        
    def load_stats(self):
        """Load statistics from the dataset."""
        train_stats, val_stats, test_stats = load_stats(self.dataset, self.device)
        self.mean_vec_x_train, self.std_vec_x_train, self.mean_vec_edge_train, self.std_vec_edge_train, self.mean_vec_y_train, self.std_vec_y_train = train_stats
        self.mean_vec_x_val, self.std_vec_x_val, self.mean_vec_edge_val, self.std_vec_edge_val, self.mean_vec_y_val, self.std_vec_y_val = val_stats
        self.mean_vec_x_test, self.std_vec_x_test, self.mean_vec_edge_test, self.std_vec_edge_test, self.mean_vec_y_test, self.std_vec_y_test = test_stats

    def rollout(self, batch: Data, batch_idx: int) -> Tuple[List[Data], List[Data], List[Data]]:
        """Rollout trajectory."""
        self.load_stats()

        data_list_true = []
        data_list_prediction = []
        data_list_error = []

        x_sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        edge_sizes = batch.n_edges.tolist()
        cells_sizes = batch.n_cells.tolist()
        i = 0
        for x, edge_index, edge_attr, y, mesh_pos, cells in zip(batch.x.split(x_sizes), torch.transpose(batch.edge_index, 0, 1).split(edge_sizes), batch.edge_attr.split(edge_sizes), batch.y.split(x_sizes), batch.mesh_pos.split(x_sizes), batch.cells.split(cells_sizes)):
            pred = self(Data(x=x, edge_index=torch.transpose(edge_index, 0, 1)-np.sum(np.array(x_sizes)[:i]), edge_attr=edge_attr), split='test')
            pred = unnormalize(data=pred, mean=self.mean_vec_y_test, std=self.std_vec_y_test)

            v = x[:, 0:2]
            if (i==0):
                prediction = v

            true = copy.deepcopy(v)
            prediction = copy.deepcopy(v)
            error = copy.deepcopy(v)

            true = v + y * self.dt
            prediction += pred * self.dt
            error = prediction - true

            data_list_true.append(Data(x=true, mesh_pos=mesh_pos, cells=cells))
            data_list_prediction.append(Data(x=prediction))
            data_list_error.append(Data(x=error))

            i += 1

        return data_list_true, data_list_prediction, data_list_error
    
    def v_noise(self, batch: Data, noise_std: float) -> torch.Tensor:
        """Return noise to add to the velocity field."""
        v = batch.x[:,:2]
        v_noise = torch.normal(std=noise_std, mean=0.0, size=v.shape).to(self.device)
        mask = torch.argmax(batch.x[:,2:],dim=1)==torch.tensor(NodeType.NORMAL)
        v_noise[mask]=0
        return v_noise