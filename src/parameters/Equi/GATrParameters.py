from dataclasses import dataclass, field   
from typing import Dict, Tuple
import torch    # just for the device

""" define all the parameters of the models and the training process """

@dataclass
class Parameters:

    print('device', torch.cuda.is_available())
    """ MODEL PARAMETERS """

    inputLayer_input_channels: int =  4     # number of starting channels
    inputLayer_output_channels: int = 8     # number of first hidden channels

    hiddenLayer_channels: int = 16          # number of hidden channels of GATr block
    
    n_heads: int = 1                  # number of heads in multihead attention
    n_blocks: int = 1                       # number of GATr blocks

    outputLayer_output_channels: int = 1    # number of output channels
    

    """ TRAINING PARAMETERS """

    batch_size: int = 8
    learning_rate: float = 1e-5    # adam learning rate
    weight_decay: float = 0.001       # adam weight decay

    max_epochs: int = 10           # number of epochs
    
    load_weights: bool = False      # if True, the model will load the weights
    weights_path: str = None        # path to the weights to load

    grade_track_norm: int = 0       # the l-norm of the grad to track, None for not tracking
    gradient_clip_val: float = 0.1  # gradient clipping value,         None for not clipping

    """ DATA PARAMETERS """

    datasets_path: str = "datasets"   # path to the datasets
    load_dataset: bool = True         # if True, the model will load the dataset in the MV format
    n_samples: int = 200              # number of samples in the dataset
    n_resempling: int = 1             # number of resempling of the dataset
    sequence_len: int = n_samples * n_resempling    # number of element in a sequence
    selected_featured: Dict[str, bool] = field(
        default_factory=lambda:{
            'wss': True, 
            'pos': True, 
            'pressure': True, 
            'inlet': False, 
            'face': True
    })
    """ GENERIC """
    device: torch.device = 'cuda' 
    MV_dim: int = 16                # mutlivector dimension --> in 3D PGA: 16
