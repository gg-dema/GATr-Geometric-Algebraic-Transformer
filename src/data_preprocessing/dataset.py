" DATA MODULE FILE "
import h5py 
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


    def __init__(self):
        """ init funct """

        # updated with data-subsample and mv_conversion method
        # std: normal raw data
        # sub_sample : subsample data (still euclidean)
        # mv : data are transformed in multivect
        self.state_of_dataset = 'std'

        self.X = None
        self.y = None

        self.n_samples = None
        self.X_sampled = None
        self.y_sampled = None 

    
        self.X_mv = None
        self.y_mv = None
        self.n_features = 4 # number of selected features --> excluded inlet
        self.mv_dim = 16 # dimension of multivector in our geomAlg G(3,0,1)


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

        X_single = []
        X_bifurcating = []

        with h5py.File(paths[SINGLE], 'r') as f:
            
            # f is a dict, enumerate all the keys is equivalent
            # to iterate over the dict
            for i, sample in enumerate(f.keys()):
                data = {}
                # extract and convert raw data to np arrays and then to torch.Tensor
                # convert raw data to Tensor is extremelly slow (warning trigger)
                data['wss'] = torch.tensor(np.array(f[sample]['wss']), dtype=torch.float32)
                data['pressure'] = torch.tensor(np.array(f[sample]['pressure']), dtype=torch.float32)
                data['pos'] = torch.tensor(np.array(f[sample]['pos']), dtype=torch.float32)
                data['face'] = torch.tensor(np.array(f[sample]['face']), dtype=torch.int64)
                data['inlet_idcs'] = torch.tensor(np.array(f[sample]['inlet_idcs']), dtype=torch.int64)  

                X_single.append(data)

    
        with h5py.File(paths[BIFURCATING], 'r') as f:

            # same concept of before, f is a dict 
            for i, sample in enumerate(f.keys()):
                data = {}
                # extract and convert raw data to np arrays and then to torch.Tensor
                # convert raw data to Tensor is extremelly slow (warning trigger)
                data['wss'] = torch.tensor(np.array(f[sample]['wss']), dtype=torch.float32)
                data['pressure'] = torch.tensor(np.array(f[sample]['pressure']), dtype=torch.float32)
                data['pos'] = torch.tensor(np.array(f[sample]['pos']), dtype=torch.float32)
                data['face'] = torch.tensor(np.array(f[sample]['face']), dtype=torch.int64)
                data['inlet_idcs'] = torch.tensor(np.array(f[sample]['inlet_idcs']), dtype=torch.int64)  

                X_bifurcating.append(data)


        y_single = torch.zeros(len(X_single))
        y_bifurcating = torch.ones(len(X_bifurcating))

        # concatenate list 
        self.X = X_single + X_bifurcating 

        # store label directly as a tensor
        self.y = torch.concatenate((y_single, y_bifurcating))
 
    
    def sub_sample_data(self, n_samples: int, n_resempling:int = 1) -> None:
        """ select a sub sample of the original sequence for each patient
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
                
                # sample separate faces and points index
                idx_face = np.random.choice(len(element['face']), n_samples, replace=False)
                idx_point = np.random.choice(len(element['pos']), n_samples, replace=False)

                # select the sampled point
                sampled_point = element['pos'][idx_point]

                # select the sampled face
                sampled_face = element['face'][idx_face]                
                
                # a plane is define by his normal and the distance from the origin
                normal, distance = self._calc_normal_and_displacement_for_plane(sampled_face, element)
                
                # create a new dict with the sampled data
                sample_data_X.append({"pos": sampled_point,
                                    "normal": normal, 
                                    "plane distance": distance, 
                                    "pressure": element['pressure'][idx_point],
                                    "wss": element['wss'][idx_point],
                                    })
                
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

        self.X_mv = torch.empty(len(self.X_sampled), 
                           self.n_features * self.n_samples, 
                           self.mv_dim)

        for i, element in enumerate(self.X_sampled):

            POS, FACE, WSS, PRESS = [], [], [], []

            for j in range(self.n_samples):
                press = torch.zeros(16)
                face = torch.zeros(16)
                position = torch.zeros(16)
                wss = torch.zeros(16)
                
                # scalar 
                press[0] = element['pressure'][j]
                
                # vector 
                face[1] = element['plane distance'][j]
                face[1:4] = element['normal'][j]

                # trivector --> e123 = 1, e0ij = pos
                position[11:14] = element['pos'][j]
                position[14] = 1.0

                # translation --> scalar + bivector e0i
                wss[0] = 1.0
                wss[5:8] = 0.5 * element['wss'][j]

                POS.append(position)
                FACE.append(face)
                WSS.append(wss)
                PRESS.append(press)

            LIST_MV = POS + FACE + WSS + PRESS
            self.X_mv[i] = torch.stack(LIST_MV)

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
