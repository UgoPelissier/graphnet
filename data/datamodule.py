import lightning.pytorch as pl
import os
import os.path as osp
from utils.utils import train_val_test_split
from data.dataset import MeshDataset
from torch_geometric.loader import DataLoader

class MeshDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            dataset_name: str,
            val_size: float,
            test_size: float,
            u_0: float,
            v_0: float,
            batch_size_train: int,
            batch_size_valid: int,
            batch_size_test: int
    ) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test

        # Define the indices
        train_index, val_index, test_index = train_val_test_split(data_dir=data_dir, name=dataset_name, n=len(os.listdir(osp.join(data_dir, dataset_name, 'raw'))), val_size=val_size, test_size=test_size)
        
        self.train_ds = MeshDataset(data_dir, dataset_name, u_0, v_0, split="train", indices=train_index)
        self.valid_ds = MeshDataset(data_dir, dataset_name, u_0, v_0, split="valid", indices=val_index)
        self.test_ds = MeshDataset(data_dir, dataset_name, u_0, v_0, split="test", indices=test_index)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size_train, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size_valid, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, num_workers=8)
