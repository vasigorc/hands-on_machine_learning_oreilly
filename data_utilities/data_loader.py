import os
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd


class DataLoader:
    def __init__(self, dataset_name, url, base_dir=None, levels_up=2):
        self.dataset_name = dataset_name
        self.url = url
        if base_dir is None:
            # Use the current working directory if base_dir is not provided
            base_dir = Path(os.getcwd())
            for _ in range(levels_up):
              base_dir = base_dir.parent
        self.dataset_dir = base_dir / "datasets"
        self.tarball_path = self.dataset_dir / f"{dataset_name}.tgz"

    def download_and_extract_data(self):
        if not self.tarball_path.is_file():
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.url, self.tarball_path)
            with tarfile.open(self.tarball_path) as tarball:
                tarball.extractall(path=self.dataset_dir)

    def load_data(self, csv_filename):
        csv_path = self.dataset_dir / self.dataset_name / csv_filename
        if not csv_path.is_file():
            self.download_and_extract_data()
        return pd.read_csv(csv_path)
