from functools import partial
import sys
import os
import os.path as osp
import logging
from typing import List, Tuple, Union
from alive_progress import alive_bar

from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from torch_geometric.data import Data
from lightning.fabric.utilities.cloud_io import get_filesystem


def load_stats(
        dataset: str,
        device: torch.device
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load statistics from the dataset."""
    train_dir = osp.join(dataset, 'processed', 'stats', 'train')
    mean_vec_x_train = torch.load(osp.join(train_dir, 'mean_vec_x.pt'), map_location=device)
    std_vec_x_train = torch.load(osp.join(train_dir, 'std_vec_x.pt'), map_location=device)
    mean_vec_edge_train = torch.load(osp.join(train_dir, 'mean_vec_edge.pt'), map_location=device)
    std_vec_edge_train = torch.load(osp.join(train_dir, 'std_vec_edge.pt'), map_location=device)
    mean_vec_y_train = torch.load(osp.join(train_dir, 'mean_vec_y.pt'), map_location=device)
    std_vec_y_train = torch.load(osp.join(train_dir, 'std_vec_y.pt'), map_location=device)
    train_stats = (mean_vec_x_train, std_vec_x_train, mean_vec_edge_train, std_vec_edge_train, mean_vec_y_train, std_vec_y_train)

    val_dir = osp.join(dataset, 'processed', 'stats', 'valid')
    mean_vec_x_val = torch.load(osp.join(val_dir, 'mean_vec_x.pt'), map_location=device)
    std_vec_x_val = torch.load(osp.join(val_dir, 'std_vec_x.pt'), map_location=device)
    mean_vec_edge_val = torch.load(osp.join(val_dir, 'mean_vec_edge.pt'), map_location=device)
    std_vec_edge_val = torch.load(osp.join(val_dir, 'std_vec_edge.pt'), map_location=device)
    mean_vec_y_val = torch.load(osp.join(val_dir, 'mean_vec_y.pt'), map_location=device)
    std_vec_y_val = torch.load(osp.join(val_dir, 'std_vec_y.pt'), map_location=device)
    val_stats = (mean_vec_x_val, std_vec_x_val, mean_vec_edge_val, std_vec_edge_val, mean_vec_y_val, std_vec_y_val)

    test_dir = osp.join(dataset, 'processed', 'stats', 'test')
    mean_vec_x_test = torch.load(osp.join(test_dir, 'mean_vec_x.pt'), map_location=device)
    std_vec_x_test = torch.load(osp.join(test_dir, 'std_vec_x.pt'), map_location=device)
    mean_vec_edge_test = torch.load(osp.join(test_dir, 'mean_vec_edge.pt'), map_location=device)
    std_vec_edge_test = torch.load(osp.join(test_dir, 'std_vec_edge.pt'), map_location=device)
    mean_vec_y_test = torch.load(osp.join(test_dir, 'mean_vec_y.pt'), map_location=device)
    std_vec_y_test = torch.load(osp.join(test_dir, 'std_vec_y.pt'), map_location=device)
    test_stats = (mean_vec_x_test, std_vec_x_test, mean_vec_edge_test, std_vec_edge_test, mean_vec_y_test, std_vec_y_test)

    return train_stats, val_stats, test_stats


def normalize(
        data: Union[torch.Tensor, List[torch.Tensor]],
        mean: Union[torch.Tensor, List[torch.Tensor]],
        std: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Normalize the data."""
    if isinstance(data, list):
        return [normalize(d, m, s) for d, m, s in zip(data, mean, std)] # type: ignore
    return (data - mean) / std


def unnormalize(
        data: Union[torch.Tensor, List[torch.Tensor]],
        mean: Union[torch.Tensor, List[torch.Tensor]],
        std: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Normalize the data."""
    if isinstance(data, list):
        return [normalize(d, m, s) for d, m, s in zip(data, mean, std)] # type: ignore
    return (data * std) + mean


def get_next_version(logs: str) -> int:
    """Get the next version number for the logger."""
    log = logging.getLogger(__name__)
    fs = get_filesystem(logs)

    try:
        listdir_info = fs.listdir(logs)
    except OSError:
        log.warning("Missing logger folder: %s", logs)
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


def progressBar(
        count_value: Union[int, float],
        total: Union[int, float],
        prefix: str=''
    ) -> None:
    """Print a progress bar."""
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + ' ' * (bar_length - filled_up_Length)
    if (percentage == 100.0):
        sys.stdout.write('%s [%s] %s%s\n' %(prefix, bar, percentage, ' %'))
    else:
        sys.stdout.write('%s [%s] %s%s\r' %(prefix, bar, percentage, ' %'))
    sys.stdout.flush()


def make_animation(
        ground_truth: List[Data],
        prediction: List[Data],
        error: List[Data],
        path: str,
        name: str,
        skip: int=2,
        save_anim: bool=True
        ) -> None:
    """Input gs is a dataloader and each entry contains attributes of many timesteps."""
    print('Generating velocity fields...')
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    num_steps = len(ground_truth) # for a single trajectory
    num_frames = num_steps // skip
    def animate(num, bar):
        step = (num*skip) % num_steps
        bar()
        traj = 0

        bb_min = ground_truth[0].x[:, 0:2].min() # first two columns are velocity
        bb_max = ground_truth[0].x[:, 0:2].max() # use max and min velocity of gs dataset at the first step for both gs and prediction plots
        bb_min_evl = error[0].x[:, 0:2].min()  # first two columns are velocity
        bb_max_evl = error[0].x[:, 0:2].max()  # use max and min velocity of gs dataset at the first step for both gs and prediction plots
        count = 0

        for ax in axes:
            ax.cla()
            ax.set_aspect('equal')
            ax.set_axis_off()
            
            pos = ground_truth[step].mesh_pos 
            faces = ground_truth[step].cells
            if (count == 0):
                # ground truth
                velocity = ground_truth[step].x[:, 0:2]
                title = 'Ground truth:'
            elif (count == 1):
                velocity = prediction[step].x[:, 0:2]
                title = 'Prediction:'
            else: 
                velocity = error[step].x[:, 0:2]
                title = 'Error: (Prediction - Ground truth)'

            triang = mtri.Triangulation(pos[:, 0].cpu(), pos[:, 1].cpu(), faces.cpu())
            if (count <= 1):
                # absolute values
                
                mesh_plot = ax.tripcolor(triang, velocity[:, 0].cpu(), vmin= bb_min, vmax=bb_max,  shading='flat' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            else:
                # error: (pred - gs)/gs
                mesh_plot = ax.tripcolor(triang, velocity[:, 0].cpu(), vmin= bb_min_evl, vmax=bb_max_evl, shading='flat' ) # x-velocity
                ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
                #ax.triplot(triang, lw=0.5, color='0.5')

            ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
            # ax.color

            # if (count == 0):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20) 
            
            clb.ax.set_title('x velocity (m/s)', fontdict = {'fontsize': 20})
            count += 1
        return fig,

    # save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)
    
    if (save_anim):
        with alive_bar(total=num_steps+1) as bar:
            gs_anim = animation.FuncAnimation(fig, partial(animate, bar=bar), frames=num_frames, interval=1000)
            writergif = animation.PillowWriter(fps=10) 
            anim_path = os.path.join(path, '{}.gif'.format(name))
            gs_anim.save(anim_path, writer=writergif) # type: ignore
            plt.show(block=True)
    else:
        pass