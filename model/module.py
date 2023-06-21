from typing import Optional
import numpy as np

from model.normalization import normalize, unnormalize, get_stats
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

    def forward(self, data: Data, mean_vec_x: float, std_vec_x: float, mean_vec_edge: float, std_vec_edge: float):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr, pressure = data.x, data.edge_index, data.edge_attr, data.p

        x = normalize(x, mean_vec_x, std_vec_x)
        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)
    
    def loss(self, pred: torch.Tensor, inputs, mean_vec_y: float, std_vec_y: float) -> torch.Tensor:
        #Define the node types that we calculate loss for
        normal=torch.tensor(0)
        outflow=torch.tensor(5)

        #Get the loss mask for the nodes of the types we calculate loss for
        loss_mask=torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==normal),
                                   (torch.argmax(inputs.x[:,2:],dim=1)==outflow))

        #Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        #Find sum of square errors
        error = torch.sum((labels-pred)**2, dim=1)

        #Root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss
    
    def on_train_start(self) -> None:
        """Set up folders for validation and test sets"""
        assert self.trainer.train_dataloader is not None
        i = 0
        for batch in self.trainer.train_dataloader:
            if i == 0:
                stats = np.array(get_stats(batch))
            else:
                stats += np.array(get_stats(batch)) # type: ignore
            i += 1
        stats /= (i+1) # type: ignore
        self.mean_vec_x, self.std_vec_x, self.mean_vec_edge, self.std_vec_edge, self.mean_vec_y, self.std_vec_y = stats # type: ignore

    def training_step(self, batch, batch_idx: int):
        """Training step of the model."""
        pass

    
    def validation_step(self, batch, batch_idx: int):
        """Validation step of the model."""
        pass
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]