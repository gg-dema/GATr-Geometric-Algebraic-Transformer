import torch
import torch.nn as nn

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Utils.geometric_utility import inner_product_through_reverse


class EquiNormLayer(nn.Module):
    def __init__(self):
        super(EquiNormLayer, self).__init__()

    def channel_expectation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation over the channel for the inner product of the input
        """
        return torch.mean(inner_product_through_reverse(x, x), dim=-1, keepdim=True)

    def forward(self, x):
        norm_squared = self.channel_expectation(x)
        # reshape for match the relative shape
        norm_squared = norm_squared.view(x.shape[0], x.shape[1], 1, 1)
        return x / torch.sqrt(norm_squared)


if __name__ == "__main__":

    from src.Utils.equivariance_check import equivariance_check

    layer_norm = EquiNormLayer()
    x = torch.randn([32, 50, 4, 16])
    y_t, t_y = equivariance_check(
        x, layer_norm, return_val=True, transformation_name="translation"
    )
    delta_translation = y_t - t_y

    print(" TRANSLATION ")
    print("max distance: ", torch.abs(delta_translation).max())
    print("min distance: ", torch.abs(delta_translation).min())
    print(
        f"mean distance: {(delta_translation).mean()}, std_dev {(delta_translation).std()}"
    )
    print(
        f"68% confidence interval:  {(delta_translation).mean() - (delta_translation).std()} / {(delta_translation).mean() + (delta_translation).std()}"
    )
