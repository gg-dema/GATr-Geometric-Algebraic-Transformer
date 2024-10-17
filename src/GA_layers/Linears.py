"""
author: DEMA, gg.dema.rm.22@gmail.com
github: gg-dema
date: 2021-10-04
"""

import torch 
import torch.nn as nn 

# append src to path (for exec as a pure module)
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --------------------------------------------
from src.Utils.geometric_utility import (
    generate_grade_projection_mask,
    calc_dual,
    apply_table_product
    )

from src.Utils.equivariance_check import equivariance_check
# --------------


class EquiLinear(nn.Module):
    """ equivariant linear layer for 3D projective geometric algebra:
        Wrapper for linear layer
    """
    def __init__(self, in_channels, out_channels, scalar_bias=False):
        super().__init__()
        
        # the input features are define by the params of the linear mapping
        # 9 params in total (5 for the blade proj, 4 for the e0 geom prod)
        self.n_basis = 9

        self.in_features = in_channels          # pratically not used
        self.out_features = out_channels

        self.scalar_bias = scalar_bias
        
        
        self.weight_Mv = torch.nn.Parameter(torch.randn(self.out_features, self.in_features, self.n_basis))
        self.register_parameter('weight_Mv', self.weight_Mv)

        if self.scalar_bias:
            # summation not implemented yet 
            self.bias_MvXe0 = torch.nn.Parameter(torch.randn(1))
            self.register_parameter('bias_MvXe0', self.bias_MvXe0)
        
        # multiplication for e0 basis:
        self.original_id = torch.tensor([0, 2, 3, 4, 8, 9, 10, 14])
        self.shifted_id = torch.tensor([1, 5, 6, 7, 11, 12, 13, 15])
        self.flip_sign_id = torch.tensor([5, 6, 7, 15])

        self.grade_projection = generate_grade_projection_mask().to(torch.bool) 
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward step of equivariant linear layer

        expected input shape: (batch, seq_len, in_channels, 16)  
        output shape: (batch, seq_len, out_channels, 16)
        """ 

        # grade_proj
        blade = input.unsqueeze(-2).repeat(1, 1, 1, 5, 1)
        blade[:, :, :, self.grade_projection.T == False] = 0

        blade_e0 = blade[:, :, :, :-1, :].clone()
        blade_e0 = self.multiply_for_e0(blade_e0)
        # shape : [batch, seq_len, channels_in, blade, mv]

        combined_Mv = torch.cat([blade, blade_e0], dim=-2)  
        output = torch.einsum('b s c k m, C c n -> b s C m', combined_Mv, self.weight_Mv)

        return output
    
    def multiply_for_e0(self, mv: torch.Tensor) -> torch.Tensor:
        '''
        geometric product for the base e0.
        Due the simple struct of the product for the basis e0, 
        we can simple shift and change the sing of some element.
        '''
        # copy original mv
        mvXe0 = torch.zeros_like(mv)

        # multiply by e0 basis shift the position of some elements...
        mvXe0[..., self.shifted_id] = mv[..., self.original_id].clone()

        # ...and also change some sing
        mvXe0[..., self.flip_sign_id] *= -1
        return mvXe0



class Bilinear(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int):
        super(Bilinear, self).__init__()

        # why this? the output is the concatenation of 2 op,
        # if this 2 operation return the out_channels dim, the output will be 2*out_channels
        # divide by 2 to have the desired output dim
        assert out_channels % 2 == 0, "out_channels must be even"

        self.mv_dim = 16
        
        # layers :
        self.join_proj_X = EquiLinear(in_channels=in_channels, out_channels=out_channels//2)
        self.join_proj_Y = EquiLinear(in_channels=in_channels, out_channels=out_channels//2)

        self.geom_prod_proj_X = EquiLinear(in_channels=in_channels, out_channels=out_channels//2)
        self.geom_prod_proj_Y = EquiLinear(in_channels=in_channels, out_channels=out_channels//2)


    def forward(self, x):

        reference_vector = torch.mean(x, dim=[0, 1, 2])
        X_join = self.join_proj_X(x)
        Y_join = self.join_proj_Y(x)

        X_geom_prod = self.geom_prod_proj_X(x)
        Y_geom_prod = self.geom_prod_proj_Y(x)

        join = self._equiJoint(X_join, Y_join, reference_vector)
        geom_prod = apply_table_product(X_geom_prod, Y_geom_prod, 'geometric')

        return torch.cat([join, geom_prod], dim=-2)

    def _equiJoint(self, X, Y, Z):
        """
        Implement the operation Z[e0123] (X* ^ Y*)*
        ----------------
        X: torch.Tensor [batch, seq_len, channels, 16] --> multivector tensor
        Y: torch.Tensor [batch, seq_len, channels, 16] --> multivector tensor
        Z = torch.Tensor [16] -> multivector reference tensor
        """
        X_dual = calc_dual(X)
        Y_dual = calc_dual(Y)
        Z[:-1] = torch.zeros(self.mv_dim - 1)
        line_intersection = apply_table_product(X_dual, Y_dual, 'wedge')
        line_intersection_dual = calc_dual(line_intersection)
        return apply_table_product(Z, line_intersection_dual, 'geometric')
        
# --------------
# TEST 
# --------------  

if __name__=="__main__":

    import argparse
    # torch.set_grad_enabled(False) --> activate to use the test

    parser = argparse.ArgumentParser() 
    parser.add_argument("--layer", type=str, default="linear", help="layer to test [linear, bilinear]")
    args = parser.parse_args()

    if args.layer == "linear":
        x = torch.randn(32, 50, 1, 16)
        layer = EquiLinear(in_channels=1, out_channels=1)
    elif args.layer == "bilinear":
        x = torch.randn(32, 50, 2, 16)
        layer = Bilinear(in_channels=2, out_channels=2)
    

    output = layer(x)
    print("TESTED LAYER: ", args.layer)
    print('output shape', output.shape)

    
    y_t, t_y = equivariance_check(x, layer, transformation_name='translation', return_val=True)
    delta_translation = y_t - t_y
    print(" TRANSLATION ")
    print("max distance: ", torch.abs(delta_translation).max())
    print("min distance: ", torch.abs(delta_translation).min())
    print(f"mean distance: {(delta_translation).mean()}, std_dev {(delta_translation).std()}")
    print(f"68% confidence interval:  {(delta_translation).mean() - (delta_translation).std()} / {(delta_translation).mean() + (delta_translation).std()}")
    
    print("\n\nROTATION ")

    y_r, r_y = equivariance_check(x, layer, return_val=True, transformation_name='rotation')
    delta_rotation = y_r - r_y
    print("max distance: ", torch.abs(delta_rotation).max())
    print("min distance: ", torch.abs(delta_rotation).min())
    print(f"mean distance: {(delta_rotation).mean()}, std_dev {(delta_rotation).std()}")
    print(f"68% confidence interval:  {(delta_rotation).mean() - (delta_rotation).std()} / {(delta_rotation).mean() + (delta_rotation).std()}")

    print("\n\n POINT REFLECTION ")
    y_point_ref, point_ref_y = equivariance_check(x, layer, "point reflection", return_val=True)
    delta_point_reflection = y_point_ref - point_ref_y
    print("max distance: ", torch.abs(delta_point_reflection).max())
    print("min distance: ", torch.abs(delta_point_reflection).min())
    print(f"mean distance: {(delta_point_reflection).mean()}, std_dev {(delta_point_reflection).std()}")
    print(f"68% confidence interval:  {(delta_point_reflection).mean() - (delta_point_reflection).std()} / {(delta_point_reflection).mean() + (delta_point_reflection).std()}")
    
 
    print("\n\n PLANE REFLECTION ")
    y_point_ref, point_ref_y = equivariance_check(x, layer, "plane reflection", return_val=True)
    delta_plane_reflection = y_point_ref - point_ref_y
    print("max distance: ", torch.abs(delta_plane_reflection).max())
    print("min distance: ", torch.abs(delta_plane_reflection).min())
    print(f"mean distance: {(delta_plane_reflection).mean()}, std_dev {(delta_plane_reflection).std()}")
    print(f"68% confidence interval:  {(delta_plane_reflection).mean() - (delta_plane_reflection).std()} / {(delta_plane_reflection).mean() + (delta_plane_reflection).std()}")

 