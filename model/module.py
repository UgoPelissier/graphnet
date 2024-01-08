from typing import Optional, List, Tuple, Union
import os
import os.path as osp
import meshio
import shutil

from graphnet.utils.stats import load_stats, normalize, unnormalize
from graphnet.utils.utils import get_next_version
from graphnet.data.dataset import NodeType
from graphnet.model.processor import ProcessorLayer

import torch
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER

from torch_geometric.data import Data

import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

class GraphNet(pl.LightningModule):
    """Lightning module for the MeshNet model."""
    def __init__(
            self,
            dir: str,
            wdir: str,
            data_dir: str,
            logs: str,
            dim: int,
            num_layers: int,
            hidden_dim: int,
            optimizer: OptimizerCallable,
            lr_scheduler: Optional[LRSchedulerCallable] = None
        ) -> None:
        super().__init__()

        self.dir = dir
        self.wdir = wdir
        self.data_dir = data_dir
        self.logs = logs
        self.dim = dim
        self.num_layers = num_layers

        # encoder convert raw inputs into latent embeddings
        input_dim_node = dim + NodeType.SIZE
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        input_dim_edge = dim + 1
        self.edge_encoder = Sequential(Linear(input_dim_edge, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))


        self.processor = torch.nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))

        # global feature
        self.global_feature = Linear((self.num_layers+1)*hidden_dim, 1024)

        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(1024+(self.num_layers+1)*hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, 1))

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.version = f'version_{get_next_version(self.logs)}'
        
    def build_processor_model(self):
        return ProcessorLayer

    def forward(
            self,
            batch: Data,
            split: str,
            mean_vec_x_predict: Optional[torch.Tensor] = None,
            mean_vec_edge_predict: Optional[torch.Tensor] = None,
            std_vec_x_predict: Optional[torch.Tensor] = None,
            std_vec_edge_predict: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr

        if split == 'train':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_train, self.mean_vec_edge_train], std=[self.std_vec_x_train, self.std_vec_edge_train])
        elif split == 'val':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_val, self.mean_vec_edge_val], std=[self.std_vec_x_val, self.std_vec_edge_val])
        elif split == 'test':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[self.mean_vec_x_test, self.mean_vec_edge_test], std=[self.std_vec_x_test, self.std_vec_edge_test])
        elif split == 'predict':
            x, edge_attr = normalize(data=[x, edge_attr], mean=[mean_vec_x_predict, mean_vec_edge_predict], std=[std_vec_x_predict, std_vec_edge_predict]) # type: ignore
        else:
            raise ValueError(f'Invalid split: {split}')

        # step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        v = x
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
            v = torch.cat((v,x),dim=1)

        # step 3: max pooling over all nodes to obtain a graph embedding
        w = v.max(dim=0, keepdim=True)[0]

        # step 4: global feature mlp
        w = self.global_feature(w)
        w = w.repeat(x.shape[0],1)
        w = torch.cat((w,v),dim=1)

        # step 5: decode latent node embeddings into physical quantities of interest
        return self.decoder(w)
    
    def loss(self, pred: torch.Tensor, inputs: Data, split: str) -> torch.Tensor:
        """Calculate the loss for the given prediction and inputs."""
        # get the loss mask for the nodes of the types we calculate loss for
        loss_mask = (torch.argmax(inputs.x[:,:NodeType.SIZE],dim=1)==torch.tensor(NodeType.NORMAL)) + (torch.argmax(inputs.x[:,:NodeType.SIZE],dim=1)==torch.tensor(NodeType.OBSTACLE)) + (torch.argmax(inputs.x[:,:NodeType.SIZE],dim=1)==torch.tensor(NodeType.OUTFLOW))
        # loss_mask = (torch.argmax(inputs.x[:,:NodeType.SIZE],dim=1)==torch.tensor(NodeType.NORMAL)) + (torch.argmax(inputs.x[:,:NodeType.SIZE],dim=1)==torch.tensor(NodeType.LOAD)) + (torch.argmax(inputs.x[:,:NodeType.SIZE],dim=1)==torch.tensor(NodeType.WALLS))

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
        error = torch.sum((labels.unsqueeze(dim=-1)-pred)**2, dim=1)

        # root and mean the errors for the nodes we calculate loss for
        loss = torch.sqrt(torch.mean(error[loss_mask]))
        
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
    
    def on_test_start(self) -> None:
        """Create the output folder for the test."""
        self.load_stats()
        os.makedirs(os.path.join(self.logs, self.version, 'test'), exist_ok=True)
        # os.makedirs(os.path.join(self.logs, self.version, 'test', 'tmp'), exist_ok=True)

    def test_step(self, batch: Data, batch_idx: int) -> None:
        """Test step of the model."""
        os.makedirs(os.path.join(self.logs, self.version, 'test', batch.name[0]), exist_ok=True)
        
        pred = self(batch, split='train')
        loss = self.loss(pred, batch, split='train')
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        pred = unnormalize(
            data=pred,
            mean=self.mean_vec_y_train,
            std=self.std_vec_y_train
        )

        point_data={
            'u_0': batch.v_0[:,0].cpu().numpy(),
            'v_0': batch.v_0[:,1].cpu().numpy(),
            'm11': batch.y.cpu().numpy(),
            'm11_pred': pred.detach().cpu().numpy()
        }

        mesh = meshio.Mesh(
            points=batch.mesh_pos.cpu().numpy(),
            cells={"tetra": batch.cells.cpu().numpy()},
            point_data=point_data
        )
        mesh.write(osp.join(self.logs, self.version, 'test', batch.name[0], f'{batch.name[0]}_pred.vtu'), binary=False)

        self.write_metric(osp.join(self.logs, self.version, 'test', batch.name[0]), batch.y, 'm')
        self.write_metric(osp.join(self.logs, self.version, 'test', batch.name[0]), pred, 'm_pred')
        shutil.copy(osp.join(self.data_dir, 'msh', f'{batch.name[0]}.msh'), osp.join(self.logs, self.version, 'test', batch.name[0], f'{batch.name[0]}.msh'))
        shutil.copy(osp.join(self.data_dir, 'mesh', f'{batch.name[0]}.mesh'), osp.join(self.logs, self.version, 'test', batch.name[0], f'{batch.name[0]}.mesh'))

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
        train_stats, val_stats, test_stats = load_stats(self.data_dir, self.device)
        self.mean_vec_x_train, self.std_vec_x_train, self.mean_vec_edge_train, self.std_vec_edge_train, self.mean_vec_y_train, self.std_vec_y_train = train_stats
        self.mean_vec_x_val, self.std_vec_x_val, self.mean_vec_edge_val, self.std_vec_edge_val, self.mean_vec_y_val, self.std_vec_y_val = val_stats
        self.mean_vec_x_test, self.std_vec_x_test, self.mean_vec_edge_test, self.std_vec_edge_test, self.mean_vec_y_test, self.std_vec_y_test = test_stats
    
    def write_field(self, path:str, field: torch.Tensor, name: str) -> None:
        with open(osp.join(path, f'{name}.txt'), 'w') as f:
            f.write(f'{len(field)}\t\n')
            for i in range(0, len(field), 5):
                if (i+5>len(field)):
                    r = len(field) - i
                    if r == 1:
                        f.write(f'\t{field[i]}\n')
                    elif r == 2:
                        f.write(f'\t{field[i]}\t{field[i+1]}\n')
                    elif r == 3:
                        f.write(f'\t{field[i]}\t{field[i+1]}\t{field[i+2]}\n')
                    elif r == 4:
                        f.write(f'\t{field[i]}\t{field[i+1]}\t{field[i+2]}\t{field[i+3]}\n')
                else:
                    f.write(f'\t{field[i]}\t{field[i+1]}\t{field[i+2]}\t{field[i+3]}\t{field[i+4]}\n')

    def write_metric(self, path:str, field: torch.Tensor, name: str) -> None:
        with open(osp.join(path, f'{name}.sol'), 'w') as f:
            f.write('MeshVersionFormatted 1\n\n')
            f.write('Dimension 3\n\n')
            f.write('SolAtVertices\n')
            f.write(f'{len(field)}\n')
            f.write('1 1\n')
            for i in range(len(field)):
                f.write(f'{field[i].item()}\n')