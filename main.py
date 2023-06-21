from data.datamodule import MeshDataModule
# from model.module import MGNModule
from lightning.pytorch.demos.boring_classes import DemoModel
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
                # "save_dir": "/data/users/upelissier/30-Code/graphnet/",
                "save_dir": "/home/eleve05/safran/graphnet/",
                "name": "logs/",
            },
        }

        parser.set_defaults(
            {
                # "data.data_dir": "/data/users/upelissier/30-Code/graphnet/data/",
                "data.data_dir": "/home/eleve05/safran/graphnet/data/",
                "data.dataset_name": "cylinder_flow",
                "data.field": "velocity",
                "data.history": False,
                "data.batch_size_train": 1,
                "data.batch_size_valid": 1,

                # "model.field": "velocity",
                # "model.node_feat_size": 2,
                # "model.edge_feat_size": 3,
                # "model.latent_size": 128,
                # "model.output_feat_size": 2,
                # "model.num_layers": 2,
                # "model.message_passing_steps": 15,
                # "model.lr": 1e-3,
                # "model.noise_scale": 0.02,
                # "model.noise_gamma": 1.0,
                # "model.decay_rate": 0.95,
                # "model.accumulate_step_size": 4, 

                "trainer.max_epochs": 100,
                "trainer.accelerator": "gpu",
                "trainer.devices": 1,
                "trainer.logger": logger,
                "trainer.callbacks": default_callbacks,
            },
        )

if __name__ == '__main__':
    cli = MyLightningCLI(
        model_class=DemoModel,
        datamodule_class=MeshDataModule,
        seed_everything_default=42,
    )