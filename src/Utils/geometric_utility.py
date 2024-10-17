import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Just as reference for the basis conversion:
# 0 -> -                                                 scalar
# 1, 2, 3, 4 -> e0, e1, e2, e3                           vector  
# 5, 6, 7, 8, 9, 10 -> e01, e02, e03, e12, e31, e23      bivector
# 11, 12, 13, 14-> e021, e013, e032, e123                trivector
# 15 -> e0123                                            pseudoscalar 


table = {
    "geometric": torch.load("src/Utils/guidance_matrix/geometric_product.pt").to_dense().to(torch.float32).to(device),
    "wedge": torch.load("src/Utils/guidance_matrix/outer_product.pt").to_dense().to(torch.float32).to(device)
}

# multiply for negative 1 ---> refer to the dual table
# link: TODO
dual_flip_sing = torch.tensor([1,                 # new scalar (1 elem)
                               1, 1, 1, 1,        # vector (4 elem) -> trivector
                               1, 1, 1, 1, 1, 1,  # bivector (6 elem) -> bivector
                               -1, -1, -1, -1,    # trivector (4 elem) -> - vector
                               -1]                # pseudoscalar (1 elem) -> - scalar 
).to(torch.float32).to(device)

# reverse modify the sign of bivector and trivector 
index = torch.zeros(16)
reverse_index = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
index[reverse_index] = 1
reverse_mask = index.to(torch.bool).to(device)

# possibility for future upgrade: 
# create a class with all the index and component, 
# then instanciate such class here (static fashion)
# then method have just to call the class index and apply it

def generate_grade_projection_mask() -> torch.Tensor:
    grade_proj_mask = torch.zeros(5, 16).to(device)
    grade_proj_mask[0, 0] = 1
    grade_proj_mask[1, 1:5] = 1
    grade_proj_mask[2, 5:11] = 1
    grade_proj_mask[3, 11:15] = 1
    grade_proj_mask[4, 15] = 1
    grade_proj_mask = grade_proj_mask.T
    return grade_proj_mask



def calc_dual(x:torch.Tensor) -> torch.Tensor:
    
    # reverse order of the mv
    dual = torch.flip(x, dims=[-1])
    # apply the flip sign
    return dual * dual_flip_sing


def apply_table_product(X:torch.Tensor, Y:torch.Tensor, op_name:str) -> torch.Tensor:
    """
    apply geom product table or wedge product table to two multivectors
    option for op_name: "geometric", "wedge"
    """
    assert op_name in table.keys(), f"Invalid table operation name -> given: {op_name}, possible: {table.keys()}"
    return  torch.einsum("i j k, ... j, ... k -> ... i", table[op_name], X, Y)

def inner_product(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    """
    inner product between two multivectors
    """
    XY = apply_table_product(X, Y, "geometric")
    YX = apply_table_product(Y, X, "geometric")

    return 0.5*(XY + YX)

def inner_product_through_reverse(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:

    """ compute inner product as <~x y>_0 """
    X_reversed = reverse(X)
    revX_Y = apply_table_product(X_reversed, Y, 'geometric')
    inner = revX_Y[..., 0]
    return inner
    


def reverse(X:torch.Tensor) -> torch.Tensor:

    """
    apply reverse operator on a tensor
    Expect input of shape [... 16]
    """
    X_rev = X.clone()
    X_rev[..., reverse_mask] *= -1
    return X_rev
