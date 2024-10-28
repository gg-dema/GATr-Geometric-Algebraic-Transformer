import torch.nn as nn
from torch import einsum


# append src to path (for exec as a pure module)
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from src.NonEquiModel.NonEquiLin import NonEquiMVLinear

class NonEquiChannelsAttention(nn.Module):

    def __init__(self, in_channels, out_channels, n_heads): 
        super(NonEquiChannelsAttention, self).__init__()
        
        self.n_heads = n_heads

        self.Q_proj = NonEquiMVLinear(in_channels, out_channels*n_heads)
        self.K_proj = NonEquiMVLinear(in_channels, out_channels*n_heads)
        self.V_proj = NonEquiMVLinear(in_channels, out_channels*n_heads)

        self.out_proj = NonEquiMVLinear(out_channels*n_heads, out_channels)

    def forward(self, x):

        # expected x shape :
        #[batch, seq_len, channels, mv_dim]
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        # expected Q, K, V shape :
        #[batch, seq_len, out_channels*n_heads, mv_dim]
        
        # put mv_dim in the second position
        V = V.permute(0, 3, 1, 2)
        K_T = K.permute(0, 3, 1, 2)
        Q = Q.permute(0, 3, 2, 1)
        # calc attention weight
        QK_T = Q@K_T
        A = nn.functional.softmax(QK_T, dim=-1)

        # attention_weight * value 
        out = einsum('b m c C, b m s C -> b s C m', A, V)
        
        out = self.out_proj(out)
        return out
    

if __name__ == "__main__":

    import torch

    x = torch.randn(32, 50, 4, 16)
    model = NonEquiChannelsAttention(in_channels=4,
                                     out_channels=16,
                                     n_heads=4)

    out = model(x)
    print(out.shape)
