# Graphnet

Graphnet is a Graph Neural Network (GNN) model for the prediction of the stationnary physical solution of a PDE, given the mesh of the geometry. The framework is based on the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library. The network is based on the [MeshGraphNets](https://arxiv.org/abs/2010.03409) architecture.

## Setup @Safran™

### Conda environment
```bash
module load conda
module load artifactory
mamba env create -f utils/envs/graphnet.yml
conda activate graphnet
```

### Download data
The data is available on the Safran GitLab. In a different folder than the one containing this repository, run:
```bash
git clone https://github.com/UgoPelissier/dataset
```
And follow the instructions in the README.md file. This will create `vtu` folders inside `stokes2`, `stokes3` and `stokes3adapt folders containing the CAD models.

Last step is to move the `stokes2`, `stokes3` and `stokes3adapt folders inside the `data` folder of this graphnet repository. The final structure should look like this:

```
├── callbacks
├── configs
├── data
└── data
    └── stokes2
        └── raw
            ├── cad_000.vtu
            :
            └── cad_500.vtu
    ├── stokes3
        └── raw
            ├── cad_000.vtu
            :
            └── cad_500.vtu
    ├── stokes3adapt
        └── raw
            ├── cad_000.vtu
            :
            └── cad_500.vtu
    ├── datamodule.py
    └── dataset.py
├── model
├── utils
├── __init__.py
├── .gitignore
├── main.py
└── README.md
```

### Train the model
To set the parameters, you can either change them inside the `main.py` file and train the model by running:
```bash
python main.py fit
```
or you can change them in the `configs/safran.yaml` file and train the model by running:
```bash
python main.py fit -c configs/safran.yaml
```

You can get help on the command line arguments by running:
```bash
python main.py fit --help
```

In both cases, it will create a new folder in the `logs/` folder containing the checkpoints of the model and a configuration file containing the parameters used for the training, that you can use later if you want.

### Evaluate the model
To evaluate the model training, run:
```bash
tensorboard --logdir=logs/
```

### Test the model
To test the model, run:
```bash
python main.py test -c configs/safran.yaml --ckpt_path $ckpt_path
```
where `$ckpt_path` is the path to the checkpoint file located in the `logs/version_$version/checkpoints/` folder.

It will create a new folder in the `logs/` folder containing the meshes resulting from the predictions of the model (`vtk` files).

## Setup @Ext™

### Conda environment
```bash
mamba env create -f utils/envs/graphnet_no_builds.yml
conda activate graphnet
```

### Download data
Follow the same instructions as for the Safran setup except that the data is available on the GitHub repository:
```bash
git clone https://github.com/UgoPelissier/dataset.git
```

### Train the model
To set the parameters, you can either change them inside the `main.py` file and train the model by running:
```bash
python main.py fit
```
or you can change them in the `configs/safran.yaml` file and train the model by running:
```bash
python main.py fit -c configs/safran.yaml
```

You can get help on the command line arguments by running:
```bash
python main.py fit --help
```

In both cases, it will create a new folder in the `logs/` folder containing the checkpoints of the model and a configuration file containing the parameters used for the training, that you can use later if you want.

### Evaluate the model
To evaluate the model training, run:
```bash
tensorboard --logdir=logs/
```

### Test the model
To test the model, run:
```bash
python main.py test -c configs/mines.yaml --ckpt_path $ckpt_path
```
where `$ckpt_path` is the path to the checkpoint file located in the `logs/version_$version/checkpoints/` folder.

It will create a new folder in the `logs/` folder containing the meshes resulting from the predictions of the model (`vtk` files).

## Contact

Ugo Pelissier \
\<[ugo.pelissier.ext@safrangroup.com](mailto:ugo.pelissier.ext@safrangroup.com)\> \
\<[ugo.pelissier@etu.minesparis.psl.eu](mailto:ugo.pelissier@etu.minesparis.psl.eu)\>
