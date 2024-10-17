" DATA MODULE FILE "
import os
import h5py 
import pickle
import torch 
import numpy as np

from torch.utils.data import Dataset

BIFURCATING = 0
SINGLE = 1

"""
Calc in a notebook: 
 min number of faces across all the sample: 10928
 min number of point across all the sample: 5466
 select subsample of N element for each sample. 
 The sampling procedure is repeat N times for each sample
"""
class MVDataset(Dataset):


    def __init__(self, selected_features: dict=None, load_data: bool=False, load_data_path: str=None):
        """ init funct """

        # updated with data-subsample and mv_conversion method

        # empty : no data
        # std: normal raw data
        # sub_sample : subsample data (still euclidean
        # mv : data are transformed in multivect


        self.state_of_dataset = 'empty'

        self.X = None
        self.y = None

        self.n_samples = None
        self.X_sampled = None
        self.y_sampled = None 

    
        self.X_mv = None
        self.y_mv = None

        if selected_features == None:
            
            self.selected_features = {
                'wss': True, 
                'pos': True, 
                'pressure': True, 
                'inlet_idcs': True, 
                'face': True
            }
            self.n_selected_features = 4 
            # inlet will be removed in train  
            # I choose to let 'inlet_idcs' to true for potential visualization of the data
            # or other usage of such classes 

        else:
            self.selected_features = selected_features

            # selected feauures is just a dict of bool val -> sum = check how many are True
            self.n_selected_features = sum(value for value in selected_features.values())

        self.feature_type = {
            'wss': torch.float32, 
            'pos': torch.float32, 
            'pressure': torch.float32, 
            'inlet_idcs': torch.int64, 
            'face': torch.int64
        }

        self.mv_dim = 16 # dimension of multivector in our geomAlg G(3,0,1)

        if load_data and load_data_path is not None:
            self.load_data(load_data_path)
            print(f'DATASET LOADED IN MV FORMAT FROM : {load_data_path}')


    def __getitem__(self, index) -> tuple:
        # the y is always the same, no change exist between the label 
        # given the different form of the input, just duplicate for the sake
        # of expleinability 

        if self.state_of_dataset == "std":
            # return list of dict of tensor, tensor
            return self.X[index], self.y[index]
        
        elif self.state_of_dataset == "sub_sample":
            # return list of dict of tensor, tensor
            return self.X_sampled[index], self.y_sampled[index]
        
        elif self.state_of_dataset == "mv": 
            # return torch tensors
            return self.X_mv[index], self.y_mv[index]
        
        else: 
            raise ValueError(f"stage of dataset not valid: {self.stage_of_dataset}")
    

    def __len__(self) -> int:

        if self.state_of_dataset == "std":
            return len(self.X)
        
        elif self.state_of_dataset == "sub_sample":
            return len(self.X_sampled)
        
        elif self.state_of_dataset == "mv":
            return len(self.X_mv)


    def load_data_files(self, paths: list) -> None:
        """ given a list of path [single path, bifurcation path], load the data"""

        assert self.state_of_dataset == 'empty', "non empty dataset, maybe you don't want ot override, no?"
        X_single = []
        X_bifurcating = []

        with h5py.File(paths[SINGLE], 'r') as f:
            
            # f is a dict, enumerate all the keys is equivalent
            # to iterate over the dict
            for i, sample in enumerate(f.keys()):
                data = {}
                # extract and convert raw data to np arrays and then to torch.Tensor
                # convert raw data to Tensor is extremelly slow (warning trigger)
                for sub_k in f[sample].keys():

                    if self.selected_features[sub_k]:
                        data[sub_k] = torch.tensor(
                            np.array(f[sample][sub_k]),
                            dtype=self.feature_type[sub_k]
                            )
                        
                X_single.append(data)

    
        with h5py.File(paths[BIFURCATING], 'r') as f:

            # same concept of before, f is a dict 
            for i, sample in enumerate(f.keys()):
                data = {}
                
                for sub_k in f[sample].keys():
                    if self.selected_features[sub_k]:
                        data[sub_k] = torch.tensor(
                            np.array(f[sample][sub_k]),
                            dtype=self.feature_type[sub_k]
                            )
                        
                X_bifurcating.append(data)


        y_single = torch.zeros(len(X_single))
        y_bifurcating = torch.ones(len(X_bifurcating))

        # concatenate list 
        self.X = X_single + X_bifurcating 

        # store label directly as a tensor
        self.y = torch.concatenate((y_single, y_bifurcating))
        self.state_of_dataset = 'std'
 
    
    def sub_sample_data(self, n_samples: int, n_resempling:int=1) -> None:
        """ 
        select a sub sample of the original sequence for each patient
        if n_resempling > 1 : sample multiple data from the same patient (kinda augemntation)
        Then, calc the geometric quantity needed for next stage
        """

        if self.state_of_dataset != "std":
            raise ValueError("Dataset has been sub sampled yet")


        sample_data_X = []
        sample_data_y = []
        # store the len of each sequence --> n sampled data in each field 
        self.n_samples = n_samples

        for i, element in enumerate(self.X):

            # equivalent to data augmentation: sample different element for the same patient 
            # each sample should be different, but it will contains the same real world obejct
            for _ in range(n_resempling):

                x = {}
                idx_point = None
                idx_face = None
                # sample separate faces and points index
                if self.selected_features['pos']:
                    idx_point = np.random.choice(len(element['pos']), n_samples, replace=False)
                    sampled_point = element['pos'][idx_point]
                    x['pos'] = sampled_point

                if self.selected_features['face']:
                    idx_face = np.random.choice(len(element['face']), n_samples, replace=False)
                    sampled_face = element['face'][idx_face]                
                    normal, distance = self._calc_normal_and_displacement_for_plane(sampled_face, element)
                    x['normal'] = normal
                    x['plane distance'] = distance 

                if self.selected_features['pressure']:
                    if idx_point is not None: # just in case we didn't sample the points yet
                        idx_point = np.random.choice(len(element['pos']), n_samples, replace=False)
                    x['pressure'] = element['pressure'][idx_point]

                if self.selected_features['wss']:
                    if idx_point is not None: # just in case we didn't sample the points yet
                        idx_point = np.random.choice(len(element['pos']), n_samples, replace=False)
                    x['wss'] = element['wss'][idx_point]

                # create a new dict with the sampled data
                sample_data_X.append(x)
                sample_data_y.append(self.y[i])

        self.X_sampled = sample_data_X
        self.y_sampled = torch.stack(sample_data_y)

        self.X = None
        self.y = None
        self.state_of_dataset = "sub_sample"


    def convert_to_mv(self) -> None:
        """
        convert subsample geometric entity into multivectors
        """

        if self.state_of_dataset != "sub_sample":
            raise ValueError("Data not sampled yet")

        self.X_mv = torch.empty(len(self.X_sampled),        # number of sample --> n_sample*seq_len
                                self.n_samples,             # number of sampled element -> seq_len
                                self.n_selected_features,   # number of channels
                                self.mv_dim)                # dimension of the multivector = 16

        for i, element in enumerate(self.X_sampled):

            # not all list used, depends on dict: selected_features
            POS, FACE, WSS, PRESS = [], [], [], []

            for j in range(self.n_samples):

                if self.selected_features['pos']:
                    # trivector --> e123 = 1, e0ij = pos
                    position = torch.zeros(16)
                    position[11:14] = element['pos'][j]
                    position[14] = 1.0
                    POS.append(position)
                
                if self.selected_features['pressure']:
                    # scalar 
                    press = torch.zeros(16)
                    press[0] = element['pressure'][j]
                    PRESS.append(press)

    
                if self.selected_features['face']:
                    # vector 
                    face = torch.zeros(16)
                    face[1] = element['plane distance'][j]
                    face[1:4] = element['normal'][j]
                    FACE.append(face)

                if self.selected_features['wss']:
                    # translation --> scalar + bivector e0i
                    wss = torch.zeros(16)
                    wss[0] = 1.0
                    wss[5:8] = 0.5 * element['wss'][j]
                    WSS.append(wss)

            
            if POS:
                POS = torch.stack(POS).unsqueeze(-2)
            if FACE:
                FACE = torch.stack(FACE).unsqueeze(-2)
            if WSS:
                WSS = torch.stack(WSS).unsqueeze(-2)
            if PRESS:
                PRESS = torch.stack(PRESS).unsqueeze(-2)

            selected_list = [
                tensor for tensor in [POS,FACE, WSS, PRESS] if isinstance(tensor, torch.Tensor)
            ]
            self.X_mv[i] = torch.concatenate(selected_list, dim=-2)

        self.y_mv = self.y_sampled

        self.X_sampled = None
        self.y_sampled = None
        self.state_of_dataset = 'mv'


    def _calc_normal_and_displacement_for_plane(self, selected_face, element) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calc normal and displacement for a plane: convert face to plane
        """
        # extract 3d point for each selected face
        point_of_face = element['pos'][selected_face]

        # calculate vector in the plane
        AB = point_of_face[:, 1, :] - point_of_face[:, 0, :]
        AC = point_of_face[:, 2, :] - point_of_face[:, 0, :]

        # cross product to get the normal, then normalize
        normal = torch.linalg.cross(AB, AC)
        normal = normal / torch.linalg.norm(normal)

        # displacement from the origin =  mean of the 3 points in Euclidean space
        displacement = torch.linalg.norm(torch.mean(point_of_face, dim=1), dim=-1)

        return normal, displacement

    def save_data(self, destination_path):

        assert self.state_of_dataset == 'mv', "Data not in MV format cannot be saved"
        # save tensors
        torch.save(self.X_mv, destination_path + 'X_mv.pt')
        torch.save(self.y_mv, destination_path + 'y_mv.pt')

        # save class metadata
        meta_data = {
            'selected features': self.selected_features,
            'n_sample': self.n_samples,
        }
        with open(os.path.join(destination_path, 'meta_data.pkl'), 'wb') as f:
            pickle.dump(meta_data, f)
        

    def load_data(self, source_path):
        # load tensors
        self.X_mv = torch.load(source_path + 'X_mv.pt')
        self.y_mv = torch.load(source_path + 'y_mv.pt')
        self.state_of_dataset = 'mv'

        with open(os.path.join(source_path, 'meta_data.pkl'), 'rb') as f:
            meta_data = pickle.load(f)
            self.selected_features = meta_data['selected features']
            self.n_samples = meta_data['n_sample']
        



if __name__ == "__main__":

    import os 
    datapath = '/home/dema/Project/GAT/datasets/'
    datapaths = [os.path.join(
                datapath, 'train', file) 
                for file in os.listdir(os.path.join(datapath, 'train'))
            ]
    
    
            
    features = {
                'wss': False, 
                'pos': True, 
                'pressure': True, 
                'inlet_idcs': False, 
                'face': True
    }

    dataset = MVDataset(selected_features=features)
    print('loading file ...')
    dataset.load_data_files(datapaths)
    print('done! \nsubsampling ...')
    dataset.sub_sample_data(2, 1)
    print('done! \nconverting to MV ...')
    dataset.convert_to_mv()
    print('done! ')

    dataset.save_data('/home/dema/Project/GAT/datasets/save_test/')
    dataset = MVDataset(load_data=True, 
                        load_data_path='/home/dema/Project/GAT/datasets/save_test/')