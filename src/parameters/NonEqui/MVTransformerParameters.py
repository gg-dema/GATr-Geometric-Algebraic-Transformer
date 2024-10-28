
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

import torch  # just for the device annotation 
              # maybe not the best strategy

@dataclass
class MVTransformerParameters:


    """ MODEL PARAMETERS """

    hidden_channels: int = 4     # number of starting channels
    
    n_heads: int = 4                        # number of heads in multihead attention
    n_blocks: int = 2                       # number of GATr blocks

    normLayer_dim : Tuple[int, int] = (16, hidden_channels)
    
    """ TRAINING PARAMETERS """

    batch_size: int = 8
    learning_rate: float = 0.001    # adam learning rate
    weight_decay: float = 0.0       # adam weight decay

    max_epochs: int = 10            # number of epochs
    
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
    device: torch.device = device   # device automatically setted 
    MV_dim: int = 16                # mutlivector dimension --> in 3D PGA: 16


MVTransf_parameters = MVTransformerParameters() 