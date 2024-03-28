# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 Nvidia
# Released under BSD-3 license https://github.com/NVIDIA/apex/blob/master/LICENSE
import glob
import os

from PIL import Image
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

from vis import misc_functions
from vis.guided_backprop import GuidedBackprop
from vis.misc_functions import save_gradient_images, convert_to_grayscale, get_positive_negative_saliency

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
        # print('model._modules: ', model._modules)

    # img_path = './raw_images/real/n01531178_1948_n01531178.JPEG'
    # img_path = './raw_images/real/n01440764_2690_n01440764.JPEG'
    # img_path = './raw_images/real/n04376876_122_n04376876.JPEG'
    # target_class = 0

    # img_path = './raw_images/fake/005_9535.jpg'
    # img_path = './raw_images/fake/img1.jpg'
    # img_path = './raw_images/fake/img2.jpg'
    img_path = './raw_images/fake/039_3767.jpg'
    target_class = 1


    file_name_to_export = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = misc_functions.preprocess_image(original_image)

    # Guided backprop
    GBP = GuidedBackprop(model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')

    pred_output = model(prep_img.cuda())
    print('pred_output: ', pred_output)

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
