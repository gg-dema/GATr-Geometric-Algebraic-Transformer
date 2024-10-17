import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from torch.nn import GELU

class EquiGeLU(GELU):

    def __init__(self):
        super(EquiGeLU, self).__init__()
    
    def forward(self, x):
        return super(EquiGeLU, self).forward(x[..., [0]])*x
    


if __name__ == "__main__":

    import torch
    from src.Utils.equivariance_check import equivariance_check

    activation = EquiGeLU()
    x = torch.randn([32, 50, 4, 16])

    y_t, t_y = equivariance_check(x, activation, return_val=True, transformation_name='translation')
    delta_translation = y_t - t_y

    print(" TRANSLATION ")
    print("max distance: ", torch.abs(delta_translation).max())
    print("min distance: ", torch.abs(delta_translation).min())
    print(f"mean distance: {(delta_translation).mean()}, std_dev {(delta_translation).std()}")
    print(f"68% confidence interval:  {(delta_translation).mean() - (delta_translation).std()} / {(delta_translation).mean() + (delta_translation).std()}")
    
