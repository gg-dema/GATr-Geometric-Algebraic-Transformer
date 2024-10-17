""" LIGHTNING DATA MODULE """

from src.data_preprocessing.dataset import MVDataset
from torch.utils.data import DataLoader, random_split

import os
import pytorch_lightning as pl


class MVDataModule(pl.LightningDataModule):

    def __init__(self,
                 datasets_path:str,
                 load_dataset:bool,
                 n_samples:int,
                 n_resempling:int,
                 batch_size: int,) -> None:
        
        super().__init__()

        self.n_samples = n_samples
        self.n_resempling = n_resempling
        self.datasets_path = os.path.join(os.getcwd(), datasets_path)
        self.batch_size = batch_size
        self.load_dataset = load_dataset

    def prepra_data(self):
        """ not used """
        pass

    def setup(self, stage=None):  
                    
        
        if stage == "fit" or stage is None:
            
            if self.load_dataset:
                # let the class dataset load the data from X_pt, y_pt, meta_data.pkl file
                self.train_dataset = MVDataset(load_data=True, 
                                               load_data_path=os.path.join(self.datasets_path, 'train/')
                                               )
                self.n_samples = self.train_dataset.n_samples

            else:
                paths_train = [os.path.join(
                    self.datasets_path, 'train', file) 
                    for file in os.listdir(os.path.join(self.datasets_path, 'train'))
                ]

                # FULL DATASET
                self.train_dataset = MVDataset()
                self.train_dataset.load_data_files(paths_train)
                self.train_dataset.sub_sample_data(self.n_samples, self.n_resempling) 
                self.train_dataset.convert_to_mv()

            # OVERWRITE THE FULL DATASET --> TRAIN / VAL
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [0.8, 0.2])
        
        if stage == "test" or None:
            
            if self.load_dataset:
                self.test_dataset = MVDataset(load_data=True,
                                              load_data_path=os.path.join(self.datasets_path, 'test/')
                                              )
                self.n_samples = self.test_dataset.n_samples
            else:
                paths_test = [os.path.join(
                    self.datasets_path, 'test', file) 
                    for file in os.listdir(os.path.join(self.datasets_path, 'train'))
                ]

                self.test_dataset = MVDataset()
                self.test_dataset.load_data_files(paths=paths_test) 
                self.test_dataset.sub_sample_data(self.n_samples, self.n_resempling)
                self.test_dataset.convert_to_mv()


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
