from data.datamodule import MeshDataModule
from model.module import MeshGraphNet

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

import warnings
warnings.filterwarnings("ignore")

class MyLightningCLI(LightningCLI):
    """Custom Lightning CLI to define default arguments."""
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        default_callbacks = [
            {
                "class_path": "callbacks.modelsummary.MyRichModelSummary",
            },
            {
                "class_path": "callbacks.progressbar.MyProgressBar",
            }
        ]

        logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": "/data/users/upelissier/30-Code/graphnet/",
                # "save_dir": "/home/eleve05/safran/graphnet/",
                "name": "logs/"
            },
        }

        parser.set_defaults(
            {
                "data.data_dir": "/data/users/upelissier/30-Code/graphnet/data/",
                # "data.data_dir": "/home/eleve05/safran/graphnet/data/",
                "data.dataset_name": "cylinder_flow",
                "data.field": "velocity",
                "data.time_steps": 600,
                "data.idx_lim_train": 4,
                "data.idx_lim_val": 1,
                "data.time_step_lim": 100,
                "data.batch_size_train": 4,
                "data.batch_size_valid": 4,

                "model.path": "/home/upelissier/30-Code/graphnet/",
                "model.dataset": "/data/users/upelissier/30-Code/graphnet/data/",
                "model.logs": "/data/users/upelissier/30-Code/graphnet/logs/",
                # "model.path": "/home/eleve05/safran/graphnet/",
                # "model.dataset": "/home/eleve05/safran/graphnet/data/",
                # "model.logs": "/home/eleve05/safran/graphnet/logs/",
                "model.num_layers": 10,
                "model.input_dim_node": 11,
                "model.input_dim_edge": 3,
                "model.hidden_dim": 10,
                "model.output_dim": 2,
                "model.optimizer": "torch.optim.AdamW",
                "model.time_step_lim": 100,
                "model.animate": True,

                "trainer.max_epochs": 1000,
                "trainer.accelerator": "gpu",
                "trainer.devices": 1,
                "trainer.check_val_every_n_epoch": 1,
                "trainer.log_every_n_steps": 1,
                "trainer.logger": logger,
                "trainer.callbacks": default_callbacks,
            },
        )

if __name__ == '__main__':
    """https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d"""
    cli = MyLightningCLI(
        model_class=MeshGraphNet,
        datamodule_class=MeshDataModule,
        seed_everything_default=42,
    )