import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_channels, output_size, layer_num, mode):
        super(Mlp, self).__init__()
        input_size = (1, in_channels)
        # input_test = torch.ones(input_size)
        self.layer_seq = nn.Sequential()
        for i in range(layer_num - 1):  # minus 1 for output_layer
            self.layer_seq.add_module(f'lmlp_{i}', nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU()))

        if mode == 'classification':
            self.layer_seq.add_module('out_mlp', nn.Sequential(
                nn.Linear(in_channels, output_size),
                nn.Softmax()))
        elif mode == 'regression':
            self.layer_seq.add_module(
                'out_mlp', nn.Linear(in_channels, output_size))

        # print("output_tensor size : ",self.layer_seq(input_test))

    def forward(self, x):
        x = self.layer_seq(x)
        return x
