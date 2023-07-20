from graphnet.data.datamodule import MeshDataModule
from graphnet.model.module import MeshGraphNet

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
                "data.data_dir": "/data/users/upelissier/30-Code/graphnet/data/stokes/",
                # "data.data_dir": "/home/eleve05/safran/graphnet/data/stokes/uniform/",
                "data.val_size": 0.15,
                "data.test_size": 0.1,
                "data.u_0": 1.0,
                "data.v_0": 0.0,
                "data.batch_size_train": 1,
                "data.batch_size_valid": 1,
                "data.batch_size_test": 1,

                "model.dir": "/data/users/upelissier/30-Code/",
                "model.wdir": "/gpfs_new/home/upelissier/30-Code/graphnet/",
                "model.data_dir": "/data/users/upelissier/30-Code/graphnet/data/stokes/",
                "model.logs": "/data/users/upelissier/30-Code/graphnet/logs/",
                # "model.path": "/home/eleve05/safran/graphnet/",
                # "model.data_dir": "/home/eleve05/safran/graphnet/data/stokes/uniform/",
                # "model.logs": "/home/eleve05/safran/graphnet/logs/",
                "model.noise_std": 2e-2,
                "model.num_layers": 15,
                "model.input_dim_node": 7,
                "model.input_dim_edge": 3,
                "model.hidden_dim": 128,
                "model.output_dim": 2,
                "model.optimizer": "torch.optim.AdamW",

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
    cli = MyLightningCLI(
        model_class=MeshGraphNet,
        datamodule_class=MeshDataModule,
        seed_everything_default=42,
    )