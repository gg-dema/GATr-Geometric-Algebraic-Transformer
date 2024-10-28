import torch
import torch.nn as nn


class NonEquiMVLinear(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(NonEquiMVLinear, self).__init__()
        self.MV_dim = 16
        
        self.w = torch.randn(out_channels, in_channels, self.MV_dim)
        self.bias = torch.randn(out_channels, self.MV_dim)

        self.w = nn.Parameter(self.w)
        self.bias = nn.Parameter(self.bias)

        self.register_parameter(name="weight", param=self.w)
        self.register_parameter(name="bias", param=self.bias)
    def forward(self, x):

        # expected x shape :
        #[batch, seq_len, channels, mv_dim]
        out = torch.einsum('b s c m, C c m -> b s C m', x, self.w)
        
        # add batch by broadcasting
        out = out + self.bias
        return out
    

if __name__ == "__main__":

    x = torch.randn(32, 50, 4, 16)
    model = NonEquiMVLinear(in_channels=4,
                              out_channels=7)

    out = model(x)
    print(out.shape)