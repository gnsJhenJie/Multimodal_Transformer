import torch
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from enum import Enum
import functools
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline

class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.):
    # Lambda function to calculate the LR
    def lr_lambda(it): return min_lr + (max_lr - min_lr) * \
        relative(it, stepsize) * decay**it

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start)*torch.pow(rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(
        anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(
        anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
    bs, tf = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n)


def extract_subTensor_per_batch_element(Tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if Tensor.is_cuda:
        batch_idxs = batch_idxs.to(Tensor.get_device())
        indices = indices.to(Tensor.get_device())
    return Tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_mask(src):
    # we usually padding sequence because of different length of input
    # we need to mask the nan element in sequence
    return torch.isnan(src)[:, :, 1]

def L2_norm(y_pred, labels):
    """
    :param: y_pred  [bs,lane_num, timestep, 2]
    :param: labels  [bs,timestep,2]
    :return the distance error of each lane_output [bs, lane_num]
    """
    return torch.norm(y_pred-labels, p=2, dim=-1)

def obs_violation_rate(predicted_trajs, groundtruth_trajs, map, heading_angle, channel, device):
    
    '''
    calc obstacle hit rate and using not xor to calc the violation of predict trajectories
    '''
    obs_map = torch.tensor(map.data[..., channel, :, :] / 255, dtype=torch.float)
    max_x, max_y = obs_map.size()
    viol_point = torch.tensor([],device=device)
    for i in range(predicted_trajs.size()[-2]):
        # homography
        px = torch.round(predicted_trajs[:, i, 0]*3).type(torch.LongTensor)
        py = torch.round(predicted_trajs[:, i, 1]*3).type(torch.LongTensor)
        gx = torch.round(groundtruth_trajs[:, i, 0]*3).type(torch.LongTensor)
        gy = torch.round(groundtruth_trajs[:, i, 1]*3).type(torch.LongTensor)
        if px > max_x or py > max_y:
            viol_point = torch.cat([viol_point,torch.ones(1,device=device)])
            continue
        elif gx > max_x or gy > max_y:
            print('wtf')
        if obs_map[px, py] == obs_map[gx, gy]:
            viol_point = torch.cat([viol_point,torch.zeros(1,device=device)])
        else:
            viol_point = torch.cat([viol_point,torch.ones(1,device=device)])
    viol_rate = torch.sum(viol_point)/ viol_point.size()[0]

    return viol_rate

def classification_loss(lane_label, lane_attn):
    # lane_label [256, prediction_horizon, 3] lane_attn [256, prediction_horizon, 3]
    # L_classification > 0 cross entropy p_i*log(p_hat_i), => p_hat_i using softmax
    cls_loss = torch.mul(lane_label, torch.log(lane_attn))
    cls_loss = torch.sum(cls_loss)
    cls_loss = (-1)*cls_loss

    return cls_loss


def confidence_loss(lanes_loss, lane_attn):
    # L_KLdiv > 0
    kl_loss = F.kl_div(F.softmax(lanes_loss), lane_attn)

    return kl_loss
