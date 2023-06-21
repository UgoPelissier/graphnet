import lightning.pytorch as pl
from data.dataset import MeshDataset
from torch_geometric.loader import DataLoader


class MeshDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 dataset_name: str,
                 field: str,
                 history: bool,
                 batch_size_train: int,
                 batch_size_valid: int
                 ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.field = field
        self.history = history
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        
        self.train_ds = MeshDataset(self.data_dir, self.dataset_name, self.field, self.history, split="train")
        self.valid_ds = MeshDataset(self.data_dir, self.dataset_name, self.field, self.history, split="valid")

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        batch = batch.to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size_train, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size_valid, shuffle=False, num_workers=8)
