import copy
from typing import Optional, List, Tuple, Union
import os.path as osp
import json

from utils.utils import get_next_version, make_animation
from model.processor import ProcessorLayer

import torch
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

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
            time_step_lim: int,
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


        self.processor = torch.nn.ModuleList()
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
        self.time_step_lim = time_step_lim
        self.version = f'version_{get_next_version(self.logs)}'
        
    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, batch: Data, split: str):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr

        x, edge_attr, _ = self.normalize(x=x, edge_attr=edge_attr, labels=None, split=split)

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
        # define the node types that we calculate loss for
        normal=torch.tensor(0)
        outflow=torch.tensor(5)

        # get the loss mask for the nodes of the types we calculate loss for
        loss_mask=torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==normal),
                                   (torch.argmax(inputs.x[:,2:],dim=1)==outflow))

        # normalize labels with dataset statistics
        _, _, labels = self.normalize(x=None, edge_attr=None, labels=inputs.y, split=split)

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

        with open(osp.join(self.dataset, 'raw', 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        self.dt = meta['dt']

        viz = copy.deepcopy(batch)
        gs = copy.deepcopy(batch)
        eval = copy.deepcopy(batch)

        pred = self(batch, split='test')

        # pred gives the learnt accelaration between two timsteps
        # next_vel = curr_vel + pred * delta_t  
        viz.x[:, 0:2] = batch.x[:, 0:2] + pred[:] * self.dt
        gs.x[:, 0:2] = batch.x[:, 0:2] + batch.y * self.dt
        # gs_data - viz_data = error_data
        eval.x[:, 0:2] = (viz.x[:, 0:2] - gs.x[:, 0:2])

        data_list_viz = []
        data_list_gs = []
        data_list_eval = []
        x_sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        mesh_pos_sizes = [int(batch.mesh_pos.shape[0]/(self.time_step_lim-1)) for i in range(self.time_step_lim-1)]
        cells_sizes = [int(batch.cells.shape[0]/(self.time_step_lim-1)) for i in range(self.time_step_lim-1)]
        for x_viz, x_gs, x_eval, mesh_pos, cells in zip(viz.x.split(x_sizes), gs.x.split(x_sizes), eval.x.split(x_sizes), batch.mesh_pos.split(mesh_pos_sizes), batch.cells.split(cells_sizes)):
            data_list_viz.append(Data(x=x_viz[:, 0:2], mesh_pos=mesh_pos, cells=cells))
            data_list_gs.append(Data(x=x_gs[:, 0:2]))
            data_list_eval.append(Data(x=x_eval[:, 0:2]))
    
        make_animation(data_list_gs, data_list_viz, data_list_eval, path=osp.join(self.logs, self.version), name='x_velocity', skip=1, save_anim=True, plot_variables=False)
    
    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], List[LRScheduler]]]:
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        
    def load_stats(self):
        """Load statistics from the dataset."""
        train_dir = osp.join(self.dataset, 'processed', 'stats', 'train')
        self.mean_vec_x_train = torch.load(osp.join(train_dir, 'mean_vec_x.pt'), map_location=self.device)
        self.std_vec_x_train = torch.load(osp.join(train_dir, 'std_vec_x.pt'), map_location=self.device)
        self.mean_vec_edge_train = torch.load(osp.join(train_dir, 'mean_vec_edge.pt'), map_location=self.device)
        self.std_vec_edge_train = torch.load(osp.join(train_dir, 'std_vec_edge.pt'), map_location=self.device)
        self.mean_vec_y_train = torch.load(osp.join(train_dir, 'mean_vec_y.pt'), map_location=self.device)
        self.std_vec_y_train = torch.load(osp.join(train_dir, 'std_vec_y.pt'), map_location=self.device)

        val_dir = osp.join(self.dataset, 'processed', 'stats', 'valid')
        self.mean_vec_x_val = torch.load(osp.join(val_dir, 'mean_vec_x.pt'), map_location=self.device)
        self.std_vec_x_val = torch.load(osp.join(val_dir, 'std_vec_x.pt'), map_location=self.device)
        self.mean_vec_edge_val = torch.load(osp.join(val_dir, 'mean_vec_edge.pt'), map_location=self.device)
        self.std_vec_edge_val = torch.load(osp.join(val_dir, 'std_vec_edge.pt'), map_location=self.device)
        self.mean_vec_y_val = torch.load(osp.join(val_dir, 'mean_vec_y.pt'), map_location=self.device)
        self.std_vec_y_val = torch.load(osp.join(val_dir, 'std_vec_y.pt'), map_location=self.device)

        test_dir = osp.join(self.dataset, 'processed', 'stats', 'test')
        self.mean_vec_x_test = torch.load(osp.join(test_dir, 'mean_vec_x.pt'), map_location=self.device)
        self.std_vec_x_test = torch.load(osp.join(test_dir, 'std_vec_x.pt'), map_location=self.device)
        self.mean_vec_edge_test = torch.load(osp.join(test_dir, 'mean_vec_edge.pt'), map_location=self.device)
        self.std_vec_edge_test = torch.load(osp.join(test_dir, 'std_vec_edge.pt'), map_location=self.device)
        self.mean_vec_y_test = torch.load(osp.join(test_dir, 'mean_vec_y.pt'), map_location=self.device)
        self.std_vec_y_test = torch.load(osp.join(test_dir, 'std_vec_y.pt'), map_location=self.device)
        
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
        elif split == 'test':
            if x is not None:
                x = (x-self.mean_vec_x_test)/self.std_vec_x_test
            if edge_attr is not None:
                edge_attr = (edge_attr-self.mean_vec_edge_test)/self.std_vec_edge_test
            if labels is not None:
                labels = (labels-self.mean_vec_y_test)/self.std_vec_y_test
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
        elif split == 'test':
            if x is not None:
                x = x*self.std_vec_x_test+self.mean_vec_x_test
            if edge_attr is not None:
                edge_attr = edge_attr*self.std_vec_edge_test+self.mean_vec_edge_test
            if labels is not None:
                labels = labels*self.std_vec_y_test+self.mean_vec_y_test
            return x, edge_attr, labels
        else:
            raise ValueError('Invalid split name')