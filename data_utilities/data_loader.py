import os
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from data_utilities.data_extensions import DataExtensions


class DataLoader:
    def __init__(
        self,
        dataset_name,
        url,
        base_dir=None,
        extension=DataExtensions.TGZ,
        levels_up=2,
    ):
        self.dataset_name = dataset_name
        self.url = url
        if base_dir is None:
            # Use the current working directory if base_dir is not provided
            base_dir = Path(os.getcwd())
            for _ in range(levels_up):
                base_dir = base_dir.parent
        self.datasets_base_dir = base_dir / "datasets"
        self.dataset_dir = self.datasets_base_dir / self.dataset_name
        self.tarball_path = self.datasets_base_dir / f"{dataset_name}{extension}"

    def download_and_extract_data(self):
        if not self.tarball_path.is_file():
            self.datasets_base_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.url, self.tarball_path)
            with tarfile.open(self.tarball_path) as tarball:
                tarball.extractall(path=self.datasets_base_dir)

    def load_data(self, csv_filename):
        csv_path = self.dataset_dir / csv_filename
        if not csv_path.is_file():
            self.download_and_extract_data()
        return pd.read_csv(csv_path)

    def is_dataset_dir_empty(self):
        if not self.dataset_dir.exists():
            return True
        return not any(self.dataset_dir.iterdir())
    
    def get_dataset_dir(self):
        return self.dataset_dir
