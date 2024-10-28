import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_dim, seq_len, device='cpu'):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pos_encoding = torch.zeros(seq_len, emb_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = pos_encoding.unsqueeze(0).to(device)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :].unsqueeze(2)  
        return x



if __name__ == "__main__":

    # Example usage
    batch_size, seq_len, channels, emb_dim = 32, 10, 8, 16
    input_tensor = torch.randn(batch_size, seq_len, channels, emb_dim)

    # Instantiate and apply positional encoding
    pos_enc = PositionalEncoding(emb_dim=emb_dim, seq_len=seq_len)
    output_tensor = pos_enc(input_tensor)

    print(output_tensor.shape)