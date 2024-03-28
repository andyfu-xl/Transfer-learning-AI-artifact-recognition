# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 Nvidia
# Released under BSD-3 license https://github.com/NVIDIA/apex/blob/master/LICENSE
import glob
import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import sys
import torch
import torch.nn as nn
from imagenet1k_sub_dataset import imageNet1kSubDataset
from tqdm import tqdm
import numpy as np
from metrics import get_classification_scores
from model import ConvolutionModel, VanillaModel
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights, resnet18
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from sklearn.exceptions import UndefinedMetricWarning
import warnings

from vis.cnn_layer_visualization import CNNLayerVisualization

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", UndefinedMetricWarning)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def run(mode, checkpoint_path=None):
    torch.cuda.set_device("cuda:0")
    if (mode == "resnext"):
        print('==> Start vis resnet...')
        weights = ResNeXt101_64X4D_Weights.DEFAULT
        model = resnext101_64x4d(weights=weights).cuda()
        for param in model.parameters():
            param.requires_grad = False
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, 2)
        # train the last conv layer
        for param in model.layer4.parameters():
            param.requires_grad = True
        # change the fc layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 2)
        )
        model.fc = model.fc.cuda()
        model = model.cuda()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
    elif (mode == "vanilla"):
        print('==> Start vis conv...')
        print(mode)
        model = VanillaModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif (mode == "convolution"):
        print('==> Start vis conv...')
        print(mode)
        model = ConvolutionModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif (mode == "resnet18"):
        print('==> Start vis resnet18 from scratch')
        print(mode)
        model = resnet18().cuda()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model.fc = model.fc.cuda()
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # change optimizer here
    else:
        raise Exception("No such mode: " + mode)

    if checkpoint_path is not None:
        model, optimizer, epoch, train_loss = load_checkpoint(checkpoint_path, model, optimizer)
        print('load checkpoint from: ', checkpoint_path)
        print('model._modules: ', model._modules)

    for cnn_layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for filter_pos in range(64):
            layer_vis = CNNLayerVisualization(model, cnn_layer, filter_pos)
            layer_vis.visualise_layer_with_hooks()
            # layer_vis.visualise_layer_without_hooks()


if __name__ == "__main__":
    # I used wandb for logging and visualization
    # it helps lots when doing remote monitoring and hyperparameter tuning
    mode = sys.argv[1]
    try:
        checkpoint_path = sys.argv[2]
        display_name = mode + '_resume_from_checkpoint'
    except IndexError:
        raise Exception("Please provide checkpoint path")
    run(mode, checkpoint_path)
