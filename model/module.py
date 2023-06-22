from typing import Optional
import os.path as osp

from model.processor import ProcessorLayer

import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
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
            num_layers: int,
            input_dim_node: int,
            input_dim_edge: int,
            hidden_dim: int,
            output_dim: int,
            optimizer: OptimizerCallable,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()

        self.path = path
        self.dataset = dataset
        self.logs = logs
        self.num_layers = num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear( hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge , hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))


        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, output_dim))

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def build_processor_model(self):
        return ProcessorLayer
    
    def normalize(self, x: torch.Tensor, edge_attr: torch.Tensor, labels: torch.Tensor, split) -> torch.Tensor:
        if split == 'train':
            if x is not None:
                x = (x-self.mean_vec_x_train)/self.std_vec_x_train
            if edge_attr is not None:
                edge_attr = (edge_attr-self.mean_vec_edge_train)/self.std_vec_edge_train    
            if labels is not None:
                labels = (labels-self.mean_vec_y_train)/self.std_vec_y_train
            return x, edge_attr, labels
        elif split == 'val':
            if x is not None:
                x = (x-self.mean_vec_x_val)/self.std_vec_x_val
            if edge_attr is not None:
                edge_attr = (edge_attr-self.mean_vec_edge_val)/self.std_vec_edge_val
            if labels is not None:
                labels = (labels-self.mean_vec_y_val)/self.std_vec_y_val
            return x, edge_attr, labels
        else:
            raise ValueError('Invalid split name')
        
    def unnormalize(self, x: torch.Tensor, edge_attr: torch.Tensor, labels: torch.Tensor, split) -> torch.Tensor:
        if split == 'train':
            if x is not None:
                x = x*self.std_vec_x_train+self.mean_vec_x_train
            if edge_attr is not None:
                edge_attr = edge_attr*self.std_vec_edge_train+self.mean_vec_edge_train
            if labels is not None:
                labels = labels*self.std_vec_y_train+self.mean_vec_y_train
            return x, edge_attr, labels
        elif split == 'val':
            if x is not None:
                x = x*self.std_vec_x_val+self.mean_vec_x_val
            if edge_attr is not None:
                edge_attr = edge_attr*self.std_vec_edge_val+self.mean_vec_edge_val
            if labels is not None:
                labels = labels*self.std_vec_y_val+self.mean_vec_y_val
            return x, edge_attr, labels
        else:
            raise ValueError('Invalid split name')

    def forward(self, data: Data, split: str):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr, pressure = data.x, data.edge_index, data.edge_attr, data.p

        x, edge_attr, _ = self.normalize(x=x, edge_attr=edge_attr, labels=None, split=split)

        # step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def loss(self, pred: torch.Tensor, inputs, split: str) -> torch.Tensor:
        """Calculate the loss for the given prediction and inputs."""
        # define the node types that we calculate loss for
        normal=torch.tensor(0)
        outflow=torch.tensor(5)

        # get the loss mask for the nodes of the types we calculate loss for
        loss_mask=torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==normal),
                                   (torch.argmax(inputs.x[:,2:],dim=1)==outflow))

        # normalize labels with dataset statistics
        _, _, labels = self.normalize(x=None, edge_attr=None, labels=labels, split=split)

        # find sum of square errors
        error = torch.sum((labels-pred)**2, dim=1)

        # root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss
    
    def on_train_start(self) -> None:
        """Set up folders for validation and test sets"""
        self.mean_vec_x_train = torch.load(osp.join(self.dataset, 'processed', 'train', 'stats', 'mean_vec_x.pt'))
        self.std_vec_x_train = torch.load(osp.join(self.dataset, 'processed', 'train', 'stats', 'std_vec_x.pt'))
        self.mean_vec_edge_train = torch.load(osp.join(self.dataset, 'processed', 'train', 'stats', 'mean_vec_edge.pt'))
        self.std_vec_edge_train = torch.load(osp.join(self.dataset, 'processed', 'train', 'stats', 'std_vec_edge.pt'))
        self.mean_vec_y_train = torch.load(osp.join(self.dataset, 'processed', 'train', 'stats', 'mean_vec_y.pt'))
        self.std_vec_y_train = torch.load(osp.join(self.dataset, 'processed', 'train', 'stats', 'std_vec_y.pt'))

        self.mean_vec_x_val = torch.load(osp.join(self.dataset, 'processed', 'val', 'stats', 'mean_vec_x.pt'))
        self.std_vec_x_val = torch.load(osp.join(self.dataset, 'processed', 'val', 'stats', 'std_vec_x.pt'))
        self.mean_vec_edge_val = torch.load(osp.join(self.dataset, 'processed', 'val', 'stats', 'mean_vec_edge.pt'))
        self.std_vec_edge_val = torch.load(osp.join(self.dataset, 'processed', 'val', 'stats', 'std_vec_edge.pt'))
        self.mean_vec_y_val = torch.load(osp.join(self.dataset, 'processed', 'val', 'stats', 'mean_vec_y.pt'))
        self.std_vec_y_val = torch.load(osp.join(self.dataset, 'processed', 'val', 'stats', 'std_vec_y.pt'))

    def training_step(self, batch, batch_idx: int):
        """Training step of the model."""
        pred = self(batch, split='train')
        loss = self.loss(pred, batch, split='train')
        self.log('train/loss', loss)

    def validation_step(self, batch, batch_idx: int):
        """Validation step of the model."""
        pred = self(batch, split='val')
        loss = self.loss(pred, batch, split='val')
        self.log('train/loss', loss)
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]