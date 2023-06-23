import copy
from typing import Optional
import os
import os.path as osp
import logging
import json

from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model.processor import ProcessorLayer

import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU

import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from lightning.fabric.utilities.cloud_io import get_filesystem

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

        self.version = f'version_{self.get_next_version()}'
        
    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, batch, split: str):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr, pressure = batch.x, batch.edge_index.long(), batch.edge_attr, batch.p

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
        _, _, labels = self.normalize(x=None, edge_attr=None, labels=inputs.y, split=split)

        # find sum of square errors
        error = torch.sum((labels-pred)**2, dim=1)

        # root and mean the errors for the nodes we calculate loss for
        loss= torch.sqrt(torch.mean(error[loss_mask]))
        
        return loss

    def training_step(self, batch, batch_idx: int):
        """Training step of the model."""
        pred = self(batch, split='train')
        loss = self.loss(pred, batch, split='train')
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx: int):
        """Validation step of the model."""
        if self.trainer.sanity_checking:
            self.load_stats()
        pred = self(batch, split='val')
        loss = self.loss(pred, batch, split='val')
        self.log('valid/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx: int):
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
    
        self.make_animation(gs, viz, eval, path=osp.join(self.logs, self.version), name='x_velocity', skip=1, save_anim=True, plot_variables=False)
    
    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        
    def make_animation(self, gs, pred, evl, path, name , skip = 2, save_anim = True, plot_variables = False):
        '''
        input gs is a dataloader and each entry contains attributes of many timesteps.

        '''
        print('Generating velocity fields...')
        fig, axes = plt.subplots(3, 1, figsize=(20, 16))
        num_steps = len(gs.ptr) # for a single trajectory
        num_frames = num_steps // skip
        print(num_steps)
        def animate(num):
            step = (num*skip) % num_steps
            traj = 0

            bb_min = gs[0].x[:, 0:2].min() # first two columns are velocity
            bb_max = gs[0].x[:, 0:2].max() # use max and min velocity of gs dataset at the first step for both gs and prediction plots
            bb_min_evl = evl[0].x[:, 0:2].min()  # first two columns are velocity
            bb_max_evl = evl[0].x[:, 0:2].max()  # use max and min velocity of gs dataset at the first step for both gs and prediction plots
            count = 0

            for ax in axes:
                ax.cla()
                ax.set_aspect('equal')
                ax.set_axis_off()
                
                pos = gs[step].mesh_pos 
                faces = gs[step].cells
                if (count == 0):
                    # ground truth
                    velocity = gs[step].x[:, 0:2]
                    title = 'Ground truth:'
                elif (count == 1):
                    velocity = pred[step].x[:, 0:2]
                    title = 'Prediction:'
                else: 
                    velocity = evl[step].x[:, 0:2]
                    title = 'Error: (Prediction - Ground truth)'

                triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
                if (count <= 1):
                    # absolute values
                    
                    mesh_plot = ax.tripcolor(triang, velocity[:, 0], vmin= bb_min, vmax=bb_max,  shading='flat' ) # x-velocity
                    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
                else:
                    # error: (pred - gs)/gs
                    mesh_plot = ax.tripcolor(triang, velocity[:, 0], vmin= bb_min_evl, vmax=bb_max_evl, shading='flat' ) # x-velocity
                    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
                    #ax.triplot(triang, lw=0.5, color='0.5')

                ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
                #ax.color

                #if (count == 0):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
                clb.ax.tick_params(labelsize=20) 
                
                clb.ax.set_title('x velocity (m/s)',
                                fontdict = {'fontsize': 20})
                count += 1
            return fig,

        # Save animation for visualization
        if not os.path.exists(path):
            os.makedirs(path)
        
        if (save_anim):
            gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000)
            writergif = animation.PillowWriter(fps=10) 
            anim_path = os.path.join(path, '{}_anim.gif'.format(name))
            gs_anim.save( anim_path, writer=writergif)
            plt.show(block=True)
        else:
            pass
        
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
        
    def get_next_version(self) -> int:
        """Get the next version number for the logger."""
        log = logging.getLogger(__name__)
        fs = get_filesystem(self.logs)

        try:
            listdir_info = fs.listdir(self.logs)
        except OSError:
            log.warning("Missing logger folder: %s", self.logs)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1