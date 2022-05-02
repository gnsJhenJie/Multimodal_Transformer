import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter
import csv
vdata = []
pdata = []


with open('../experiments/processed/nuScenes_train_full.pkl', 'rb') as f:
    train_env = dill.load(f, encoding='latin1')
    train_scenes = train_env.scenes
    i = 0
    # print(train_scenes)
    first = True
    for x in train_scenes:
        for node in x.nodes:
            for obj in node.data.data:
                if(len(obj) == 12):
                    if first:
                        print(node.data.header)
                        first = False
                    vdata.append(obj)
                    i += 1
            if i >= 1000:
                break
        if i >= 1000:
            break
    i = 0
    first = True
    for x in train_scenes:
        for node in x.nodes:
            for obj in node.data.data:
                if(len(obj) == 6):
                    if first:
                        print(node.data.header)
                        first = False
                    pdata.append(obj)
                    i += 1
            if i >= 1000:
                break
        if i >= 1000:
            break

with open('nuScene_v.csv', 'w') as csvf:
    writer = csv.writer(csvf)
    writer.writerows(vdata)
with open('nuScene_p.csv', 'w') as csvf:
    writer = csv.writer(csvf)
    writer.writerows(pdata)

vdata = []
pdata = []


with open('../experiments/processed/Pandaset_train_full.pkl', 'rb') as f:
    train_env = dill.load(f, encoding='latin1')
    train_scenes = train_env.scenes
    i = 0
    # print(train_scenes)
    first = True
    for x in train_scenes:
        for node in x.nodes:
            for obj in node.data.data:
                if(len(obj) == 12):
                    if first:
                        print(node.data.header)
                        first = False
                    vdata.append(obj)
                    i += 1
            if i >= 1000:
                break
        if i >= 1000:
            break
    i = 0
    first = True
    for x in train_scenes:
        for node in x.nodes:
            for obj in node.data.data:
                if(len(obj) == 6):
                    if first:
                        print(node.data.header)
                        first = False
                    pdata.append(obj)
                    i += 1
            if i >= 1000:
                break
        if i >= 1000:
            break

with open('pandaset_v.csv', 'w') as csvf:
    writer = csv.writer(csvf)
    writer.writerows(vdata)
with open('pandaset_p.csv', 'w') as csvf:
    writer = csv.writer(csvf)
    writer.writerows(pdata)
