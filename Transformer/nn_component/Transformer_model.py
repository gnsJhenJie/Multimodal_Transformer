import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class Encoder(nn.Module):
    def __init__(
            self,
            ninp=8,
            ntimestep=9,
            in_dim=128,
            nhead=2,
            fdim=256,
            nlayers=6,
            noutput=32,
            dropout=0.2,
            low_dim="sum"):
        super(Encoder, self).__init__()

        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(ninp, in_dim)
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        encoder_layers = TransformerEncoderLayer(
            in_dim, nhead, fdim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.low_dim = low_dim
        if self.low_dim == "sum":
            self.output_fc = nn.Linear(in_dim, noutput)
        elif self.low_dim == "flatten":
            self.output_fc = nn.Linear(in_dim * ntimestep, noutput)
        # self.init_weights()

    def init_weights(self) -> None:
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        torch.nn.init.zeros_(self.input_fc.bias)
        torch.nn.init.xavier_normal_(self.input_fc.weight)
        torch.nn.init.zeros_(self.output_fc.bias)
        torch.nn.init.xavier_normal_(self.output_fc.weight)

    def forward(self, src, src_mask, src_key_padding_mask):

        input = torch.nan_to_num(src, nan=0)
        input = self.input_fc(input)
        input = self.pos_encoder(input)
        memory = self.transformer_encoder(
            input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # if self.low_dim == "sum":
        #     feature = torch.sum(memory, 1)
        # elif self.low_dim == "flatten":
        #     feature = torch.flatten(memory, 1)
        # feature = self.output_fc(feature)

        return memory


class Decoder(nn.Module):
    def __init__(
            self,
            nlayers,
            ninp,
            in_dim,
            nhead,
            fdim,
            noutput,
            dropout=0.2):
        super().__init__()

        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(ninp, in_dim)
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        decoder_layers = TransformerDecoderLayer(
            in_dim, nhead, fdim, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.output_fc = nn.Linear(in_dim, noutput)
        # self.init_weights()

    def init_weights(self) -> None:
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        torch.nn.init.zeros_(self.input_fc.bias)
        torch.nn.init.xavier_normal_(self.input_fc.weight)
        torch.nn.init.zeros_(self.output_fc.bias)
        torch.nn.init.xavier_normal_(self.output_fc.weight)

    @abc.abstractmethod
    def forward(
            self,
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            memory_key_padding_mask):
        # input tgt => target last time state

        output = self.transformer_decoder(
            self.pos_encoder(
                self.input_fc(tgt)),
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_fc(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (bs,timestep,feature)
        x = x + Variable(self.pe[:x.size()[-2]], requires_grad=False)
        return self.dropout(x)


class Lane_Decoder(Decoder):
    def __init__(self, ninp, in_dim, nhead, fdim, nlayers, noutput):
        super().__init__(nlayers, ninp, in_dim, nhead, fdim, noutput)

    def forward(self, tgt, memory, memory_key_padding_mask):
        """ To inference the Decoder

        Args:
            tgt (torch.tensor): Encoded Lane Tensor (bs, lane_num, lane_feature)
            memory (torch.tensor): Encoded Node History (bs, timestep, node_feature)
            memory_key_padding_mask (torch.tensor): Memory padding mask (bs, timestep)

        Returns:
            torch.tensor: Decoded Lane feature (bs, lane_num, noutput)
        """

        output = self.transformer_decoder(self.pos_encoder(self.input_fc(
            tgt)), memory, memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_fc(output)

        return output


class Trajectory_Decoder(Decoder):
    def __init__(
            self,
            tgt_inp,
            lane_inp,
            in_dim,
            nhead,
            fdim,
            nlayers,
            noutput):
        super().__init__(nlayers, tgt_inp, in_dim, nhead, fdim, noutput)
        self.mix_fc = nn.Linear(lane_inp + in_dim, in_dim)

    def forward(
            self,
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            memory_key_padding_mask,
            lane_feature=None):
        """ To inference the Decoder

        Args:
            tgt (torch.tensor): The current position of agent
            memory ([type]): Mix Tensor or only node history
            lane_feature (torch.tensor): Decoded Lane feature
            tgt_mask (torch.tensor): Target sequence mask
            memory_key_padding_mask (torch.tensor): Memory padding mask

        Returns:
            torch.tensor: Decoded future state feature
        """
        # TODO lane_inp + indim use config to get value
        if lane_feature is not None:
            mix_Tensor = torch.cat([memory, lane_feature], dim=-1)
            memory = self.mix_fc(mix_Tensor)
        output = self.transformer_decoder(
            self.pos_encoder(
                self.input_fc(tgt)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_fc(output)
        return output
