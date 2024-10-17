"""
Simple script for split the dataset in trian and test

"""

import h5py


def split_dataset(path_original: str, 
                  path_train: str,
                  path_test: str, 
                  percent: float = 0.8):
    
    with h5py.File(path_original, 'r') as ref_f:

        # select amount of data for each file
        index = int(percent* len(ref_f.keys())) - 1

        with (
            h5py.File(path_train, 'w') as train_f,
            h5py.File(path_test, 'w') as test_f
            ):
            for i, group_name in enumerate(ref_f.keys()):
            
                if i <= index:
                    train_f.copy(ref_f[group_name], group_name)
                else:
                    test_f.copy(ref_f[group_name], group_name)



# Single
path = "datasets/original_datasets/single.hdf5"
path_test = "datasets/test/single.hdf5"
path_train = "datasets/train/single.hdf5"
split_dataset(path, path_train, path_test)

# bifurcating: 

path = "datasets/original_datasets/bifurcating.hdf5"
path_test = "datasets/test/bifurcating.hdf5"
path_train = "datasets/train/bifurcating.hdf5"
split_dataset(path, path_train, path_test)