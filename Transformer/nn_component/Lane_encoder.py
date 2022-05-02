import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Lane_Encoder(nn.Module):
    # encode raster map and output encoding tensor
    def __init__(
            self,
            layers,
            embedding_size,
            output_channels,
            output_size,
            kernel_size,
            strides):
        super(Lane_Encoder, self).__init__()
        # Using dummy input to initialize the neural networks parameter

        x_dummy = torch.zeros((1, 2, 10))
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv1d(embedding_size, output_channels, kernel_size, strides))
        x_dummy = self.convs[0](x_dummy)
        # print("x_dummy size : ", x_dummy.size())
        x_dummy = torch.flatten(x_dummy, start_dim=1)
        self.fc = nn.Linear(x_dummy.size()[-1], output_size)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)

        return self.fc(torch.flatten(x, start_dim=1))


if __name__ == "__main__":

    dictionary = {
        "lane_encoder": {
            "VEHICLE": {
                "layers": 1,
                "embedding_size": 2,
                "output_channels": 16,
                "output_size": 32,
                "kernel_size": 4,
                "strides": 2,
                "dropout": 0.5,
            }
        }
    }

    me_params = dictionary["lane_encoder"]["VEHICLE"]
    test = Lane_Encoder(
        me_params["layers"],
        me_params["embedding_size"],
        me_params["output_channels"],
        me_params["output_size"],
        me_params["kernel_size"],
        me_params["strides"],
    )
    input = torch.randn(256, 3, 2, 10).view(256 * 3, 2, 10)
    output = None
    print("Input size : ", input.size())
    output = test(input).view(256, 3, 32)
    print("Output size : ", output.size())
