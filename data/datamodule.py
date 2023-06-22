import lightning.pytorch as pl
from data.dataset import MeshDataset
from torch_geometric.loader import DataLoader

class MeshDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 dataset_name: str,
                 field: str,
                 time_steps: int,
                 idx_lim_train: int,
                 idx_lim_val: int,
                 time_step_lim: int,
                 batch_size_train: int,
                 batch_size_valid: int
                 ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.field = field
        self.time_steps = time_steps
        self.idx_lim_train = idx_lim_train
        self.idx_lim_val = idx_lim_val
        self.time_step_lim = time_step_lim
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        
        self.train_ds = MeshDataset(self.data_dir, self.dataset_name, self.field, self.time_steps, self.idx_lim_train, self.idx_lim_val, self.time_step_lim, split="train")
        self.valid_ds = MeshDataset(self.data_dir, self.dataset_name, self.field, self.time_steps, self.idx_lim_train, self.idx_lim_val, self.time_step_lim, split="valid")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size_train, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size_valid, shuffle=False, num_workers=8)
