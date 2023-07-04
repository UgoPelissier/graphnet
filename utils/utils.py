import os
import os.path as osp
import logging
import numpy as np
from lightning.fabric.utilities.cloud_io import get_filesystem
from typing import Tuple
import vtk

def train_val_test_split(
        data_processed: str,
        name: str,
        n: int,
        val_size: float,
        test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train, validation and test sets."""
    os.makedirs(osp.join(data_processed, name, 'indices'), exist_ok=True)
    indices = np.random.permutation(n)
    train_index, val_index, test_index = indices[:int(n*(1-(val_size+test_size)))], indices[int(n*(1-(val_size+test_size))):int(n*(1-test_size))],  indices[int(n*(1-test_size)):]
    np.savetxt(osp.join(data_processed, name, 'indices', 'train_index.txt'), train_index, fmt='%i')
    np.savetxt(osp.join(data_processed, name, 'indices', 'val_index.txt'), val_index, fmt='%i')
    np.savetxt(osp.join(data_processed, name, 'indices', 'test_index.txt'), test_index, fmt='%i')
    return train_index, val_index, test_index


def load_train_val_test_index(data_processed: str, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the train, validation and test sets indices."""
    return np.loadtxt(osp.join(data_processed, name, 'indices', 'train_index.txt'), dtype=int), np.loadtxt(osp.join(data_processed, name, 'indices', 'val_index.txt'), dtype=int), np.loadtxt(osp.join(data_processed, name, 'indices', 'test_index.txt'), dtype=int)


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


def cell2point(
        file: str,
        field: str
) -> np.ndarray:
    """Convert cell data to point data."""
    reader = vtk.vtkXMLUnstructuredGridReader()  
    reader.SetFileName(file) 
    reader.Update()

    converter = vtk.vtkCellDataToPointData()
    converter.ProcessAllArraysOn()
    converter.SetInputConnection(reader.GetOutputPort())
    converter.Update()
    return np.array(converter.GetOutput().GetPointData().GetArray(field))
