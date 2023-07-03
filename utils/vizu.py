from functools import partial
import os
from typing import List
from alive_progress import alive_bar

from matplotlib import pyplot as plt
from matplotlib import tri as mtri
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch_geometric.data import Data

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