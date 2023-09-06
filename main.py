from graphnet.data.datamodule import MeshDataModule
from graphnet.model.module import GraphNet

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
                "save_dir": "/data/users/upelissier/30-Code/graphnet/",  # TODO: Parent directory of graphnet folder
                "name": "logs/"
            },
        }

        parser.set_defaults(
            {
                "data.data_dir": "/data/users/upelissier/30-Code/graphnet/data/stokes/", # TODO: Data directory
                "data.dim": 3, # TODO: Dimension of the problem
                "data.val_size": 0.15, # Validation size
                "data.test_size": 0.1, # Test size
                "data.u_0": 1.0, # Boundary condition
                "data.v_0": 0.0, # Boundary condition
                "data.w_0": 0.0, # Boundary condition
                "data.batch_size_train": 1, # Batch size
                "data.batch_size_valid": 1, # Batch size
                "data.batch_size_test": 1, # Batch size

                "model.dir": "/data/users/upelissier/30-Code/", # TODO: Parent directory
                "model.wdir": "/home/upelissier/30-Code/graphnet/", # TODO: Working directory
                "model.data_dir": "/data/users/upelissier/30-Code/graphnet/data/stokes3/", # TODO: Data directory
                "model.logs": "/data/users/upelissier/30-Code/graphnet/logs/", # TODO: Logs directory
                "model.dim": 3, # TODO: Dimension of the problem
                "model.num_layers": 15, # Number of layers
                "model.hidden_dim": 128, # Hidden dimension
                "model.optimizer": "torch.optim.AdamW", # Optimizer

                "trainer.max_epochs": 1000, # TODO: Maximum number of epochs
                "trainer.accelerator": "gpu", # Accelerator
                "trainer.devices": 2, # TODO: Number of devices
                "trainer.check_val_every_n_epoch": 1, # Check validation every n epochs
                "trainer.log_every_n_steps": 1, # Log every n steps
                "trainer.logger": logger, # Logger
                "trainer.callbacks": default_callbacks, # Callbacks
            },
        )

if __name__ == '__main__':
    cli = MyLightningCLI(
        model_class=GraphNet,
        datamodule_class=MeshDataModule,
        seed_everything_default=42,
    )