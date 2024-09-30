""" LIGHTNING DATA MODULE """

from data_preprocessing.dataset import MVDataset
from torch.utils.data import DataLoader, random_split

import os
import pytorch_lightning as pl


class MVDataModule(pl.LightningDataModule):

    def __init__(self,
                 datasets_path:str, 
                 n_samples:int,
                 n_resempling:int,
                 batch_size: int,) -> None:
        
        super().__init__()

        self.n_samples = n_samples
        self.n_resempling = n_resempling
        self.datasets_path = datasets_path
        self.batch_size = batch_size

    def setup(self, stage=None):  
                    
    
        if stage == "fit" or stage is None:

            paths_train = [os.path.join(
            os.getcwd(), self.datasets_path, 'train', file)
            for file in os.listdir(self.datasets_path)
            ]

            # FULL DATASET
            self.train_dataset = MVDataset.load_data_files(paths_train)
            self.train_dataset.sub_sample_data() 
            self.train_dataset.convert_to_mv(self.n_samples, self.n_resempling)

            # OVERWRITE THE FULL DATASET --> TRAIN / VAL
            self.train_dataset, self.val_dataset = random_split(self.train_dataset_full, [0.8, 0.2])
        
        if stage == "test" or stage is None:

            paths_test = [os.path.join(
            os.getcwd(), self.datasets_path, 'test', file)
            for file in os.listdir(self.datasets_path)
            ]

            self.test_dataset = MVDataset().load_data_files(paths=paths_train) 
            self.test_dataset.sub_sample_data(self.n_samples, self.n_resempling)
            self.test_dataset.convert_to_mv()


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataloader, batch_size=self.batch_size)
