import clifford as cl
import torch 

import warnings
from numba.core.errors import NumbaWarning

# Suppress all warnings from Numba
warnings.simplefilter('ignore', category=NumbaWarning)


class EquiCheckerUtility:

    def __init__(self):
        
        # Just as reference for the basis conversion:
        # 0 -> -                                                 scalar
        # 1, 2, 3, 4 -> e0, e1, e2, e3                           vector  
        # 5, 6, 7, 8, 9, 10 -> e01, e02, e03, e12, e31, e23      bivector
        # 11, 12, 13, 14-> e021, e013, e032, e123                trivector
        # 15 -> e0123                                            pseudoscalar 

        self.generator = {
            'translation': self.generate_translation_motor,
            'rotation': self.generate_rotation_motor,
            'point reflection': self.generate_point_reflection_motor,
            'plane reflection': self.generate_plane_reflection_motor,
        }
        self.involution_required = {
            'translation': False,       # encoded as bivector -> even grade
            'rotation': False,          # encoded as bivector -> even grade
            'point reflection': True,   # encoded as trivector -> odd grade
            'plane reflection': True    # encoded as vector -> odd grade
        }

        self.rotation_id = [8, 9, 10]               # e basis for rotation                         
        self.translation_id = [5, 6, 7]             # e01, e02, e03 basis for translation          
        self.point_reflection_id = [11, 12, 13]     # e021, e031 e032 basis for point reflection   
        self.plane_reflection_id = [2, 3, 4]        #  e1, e2, e3 basis for plane reflection    |  e0 out for offset


    def generate_translation_motor(self, reference_shape: torch.Size) -> list[cl.MultiVector]:
        """
        generate a list of translation motor (MV) for each element of the batch
        """
        assert reference_shape[-1] == 16, "The last dim should be 16, AKA multivector dim"
        translation = torch.zeros(reference_shape)
        translation[..., 0] = 1    # scalar componet is 1: check the table 
        translation[..., self.translation_id] = torch.randn(3)

        T_mv = [layout.MultiVector(t) for t in translation]
        return T_mv


    def generate_rotation_motor(self, reference_shape: torch.Size) -> list[cl.MultiVector]:
        """
        generate a list of rotation motor (MV) for each element of the batch
        """

        assert reference_shape[-1] == 16, "The last dim should be 16, AKA multivector dim"
        rotation = torch.zeros(reference_shape)
        quaternion_val = torch.randn(4)
        quaternion_val = quaternion_val / torch.norm(quaternion_val, dim=-1, keepdim=True)

        rotation[..., 0] = quaternion_val[0] # scalar componet is 1: check the table 
        rotation[..., self.rotation_id] = quaternion_val[1:]

        R_mv = [layout.MultiVector(r) for r in rotation]
        return R_mv
    

    def generate_point_reflection_motor(self, reference_shape: torch.Size) -> list[cl.MultiVector]:
        """
        generate a list of point reflection motor (MV) for each element of the batch
        """

        assert reference_shape[-1] == 16, "The last dim should be 16, AKA multivector dim"
        points_mv = torch.zeros(reference_shape)
        points = torch.randn(3)

        points_mv[..., 14] = 1  # 1 for e123 basis 
        points_mv[..., self.point_reflection_id] = points

        points_mv = [layout.MultiVector(p) for p in points_mv]
        return points_mv
    

    def generate_plane_reflection_motor(self, reference_shape: torch.Size) -> list[cl.MultiVector]:
        
        assert reference_shape[-1] == 16, "The last dim should be 16, AKA multivector dim"
        planes_mv = torch.zeros(reference_shape)

        offsets = torch.randn(reference_shape[0])
        normals = torch.randn(3)
        normals /= torch.linalg.norm(normals, dim=-1, keepdim=True)

        planes_mv[..., 1] = offsets  # offset for e0 basis 
        planes_mv[..., self.plane_reflection_id] = normals
        planes_mv = [layout.MultiVector(p) for p in planes_mv]
        return planes_mv



layout, blade = cl.Cl(3, 0, 1) # define the PGA layout
eq_utility = EquiCheckerUtility()

def equivariance_check(
        x:torch.Tensor,
        layer:callable, 
        transformation_name:str, 
        return_val=False, 
        limit=1e-3, 
        device='cpu'):
    """
    Check the equivariance of a transformation respect to a SPIN group element 
    -------
    x = a torch tensor [our basic input, could be any shape until the last is the mv_dim=16],
    layer = just a (neural or not) transformation, such a torch.nn or other function
    transformation_name = the name of the transformation to check. see the list below
    return_val = if True, return the two output y = f(T(x)), y' = T(f(x)) 
    limit = a tollerance of error. Not used if return val is true

    """
    valid_transformation = ['rotation', 'translation', 'point reflection', 'plane reflection']
    assert transformation_name in valid_transformation, f"transformation_name should be one of {valid_transformation}"

    input_shape = x.shape
    flat_x = x.reshape(-1, 16)
    transformation_mv = eq_utility.generator[transformation_name](flat_x.shape)

    # FIRST PART : apply transformation and then layer: y_t = layer(T(x))
    # ---------------------------------------------
    if eq_utility.involution_required[transformation_name]:
        MV_X = [layout.MultiVector(flat_x[i, :].cpu()).gradeInvol() for i in range(flat_x.shape[0])]
    else:
        MV_X = [layout.MultiVector(flat_x[i, :].cpu()) for i in range(flat_x.shape[0])]

    # apply transformation
    MV_X_transformed = [(transformation_mv[i] * MV_X[i] * ~transformation_mv[i]) for i in range(flat_x.shape[0])]

    # come back to torch tensor:
    X_transformed = [torch.Tensor(MV_X_transformed[i].value) for i in range(flat_x.shape[0])]
    X_transformed = torch.stack(X_transformed).reshape(input_shape).to(device)
    y_of_transform = layer(X_transformed)

    # SECOND PART : apply layer and then transformation: t_y = T(layer(x))
    # ---------------------------------------------

    # project to layer
    y = layer(x)
    flat_y = y.reshape(-1, 16)

    # convert to multivector (and grade involution if needed)
    if eq_utility.involution_required[transformation_name]:
        MV_y = [layout.MultiVector(flat_y[i, :].cpu()).gradeInvol() for i in range(flat_y.shape[0])]
    else:
        MV_y = [layout.MultiVector(flat_y[i, :].cpu()) for i in range(flat_y.shape[0])]
    
    # apply transformation
    transform_of_MV_y = [transformation_mv[i] * MV_y[i] * ~transformation_mv[i] for i in range(flat_y.shape[0])]

    # come back to torch tensor:
    transform_of_y = [torch.Tensor(transform_of_MV_y[i].value) for i in range(flat_y.shape[0])]
    transform_of_y = torch.stack(transform_of_y).reshape(input_shape).to(device)
    
    if return_val:
        return y_of_transform, transform_of_y
    
    assert torch.allclose(y_of_transform, transform_of_y, atol=limit), "Equivariance check failed"
    return "equivariance to translation: CHECKED"



