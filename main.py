# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 Nvidia
# Released under BSD-3 license https://github.com/NVIDIA/apex/blob/master/LICENSE
import glob
import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import wandb
import sys
import torch
import torch.nn as nn
from imagenet1k_sub_dataset import imageNet1kSubDataset
from tqdm import tqdm
import numpy as np
from metrics import get_classification_scores
from model import ConvolutionModel, VanillaModel
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights, resnet18

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", UndefinedMetricWarning)

def train(model, optimizer, loss_fn, trainloader):
    model.train()
    with tqdm(total=len(trainloader)) as pbar:  # create a progress bar for training
        for idx, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Enables auto casting for the forward pass (model + loss)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(images)
                loss = loss_fn(pred, labels)
            # pred = model(images)
            # loss = loss_fn(pred, labels)

            # Exits the context manager before backward()
            loss.backward()
            pbar.update(1)
            pbar.set_description("loss: {:.4f}".format(loss))
            optimizer.step()

    classification_scores = get_classification_scores(pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
    wandb.log({"train/train_batch_loss": loss.item(), "train/accuracy": classification_scores['accuracy'],
               "train/precision": classification_scores['precision'], "train/recall": classification_scores['recall'],
               "train/f1": classification_scores['f1']})
    return loss

def evaluate(model, loss_fn, valloader):
    # Validation loss
    total_loss = 0.0
    epoch_steps = 0

    output_list = []
    label_list = []

    model.eval()
    with tqdm(total=len(valloader)) as pbar:  # create a progress bar for training
        for idx, (images, labels) in enumerate(valloader):
            with torch.no_grad():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                # outputs = model(images)

                output_list.append(outputs.detach().cpu().numpy())
                label_list.append(labels.detach().cpu().numpy())

                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                epoch_steps += 1
                pbar.update(1)
                pbar.set_description("loss: {:.4f}".format(loss))

    val_loss = total_loss / epoch_steps
    output_list = np.concatenate(output_list)
    label_list = np.concatenate(label_list)
    classification_scores = get_classification_scores(output_list, label_list)
    wandb.log({"val/val_loss": val_loss, "val/accuracy": classification_scores['accuracy'],
               "val/precision": classification_scores['precision'], "val/recall": classification_scores['recall'],
               "val/f1": classification_scores['f1']})
    print('val_loss: ', val_loss, 'classification_scores: ', classification_scores.items())
    return val_loss

def test_accuracy(model, loss_fn, testloader):
    # test loss
    total_loss = 0.0
    epoch_steps = 0

    output_list = []
    label_list = []

    model.eval()
    with tqdm(total=len(testloader)) as pbar:  # create a progress bar for training
        for idx, (images, labels) in enumerate(testloader):
            with torch.no_grad():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)

                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                epoch_steps += 1

                output_list.append(outputs.detach().cpu().numpy())
                label_list.append(labels.detach().cpu().numpy())
                pbar.update(1)
                pbar.set_description("loss: {:.4f}".format(loss))

    test_loss = total_loss / epoch_steps

    output_list = np.concatenate(output_list)
    label_list = np.concatenate(label_list)
    classification_scores = get_classification_scores(output_list, label_list)
    wandb.log({"test/test_loss": test_loss, "test/accuracy": classification_scores['accuracy'],
               "test/precision": classification_scores['precision'], "test/recall": classification_scores['recall'],
               "test/f1": classification_scores['f1']})
    print('test_loss: ', test_loss, 'classification_scores: ', classification_scores.items())

    return test_loss

def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def run(mode, checkpoint_path=None):
    torch.cuda.set_device("cuda:0")

    batch_size = 256

    total_epochs = 100

    # setup dataset
    print('==> Preparing data...')
    data_path = './data'
    # data_path_list = glob.glob(data_path + '/*/*/*.jpg') + glob.glob(data_path + '/*/*/*.JPEG')
    # dataset = imageNet1kSubDataset(data_path_list=data_path_list)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    dataset = ImageFolder(root=data_path, transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        ))
    print('dataset size: ', len(dataset))
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    proportions = [train_ratio, val_ratio, test_ratio]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths)
    print('train set size: ', len(train_set))
    print('val set size: ', len(val_set))
    print('test set size: ', len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    if(mode == "resnext"):
        print('==> Start training resnet...')
        weights = ResNeXt101_64X4D_Weights.DEFAULT
        model = resnext101_64x4d(weights=weights).cuda()
        for param in model.parameters():
            param.requires_grad = False
        #num_ftrs = model.fc.in_features
        #model.fc = torch.nn.Linear(num_ftrs, 2)
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
    elif(mode == "vanilla"):
        print('==> Start training conv...')
        print(mode)
        model = VanillaModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()  # change loss function here
    elif (mode == "convolution"):
        print('==> Start training conv...')
        print(mode)
        model = ConvolutionModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss() # change loss function here
    elif (mode == "resnet18"):
        print('==> Start training resnet18 from scratch')
        print(mode)
        model = resnet18().cuda()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model.fc = model.fc.cuda()
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # change optimizer here
        loss_fn = torch.nn.CrossEntropyLoss()  # change loss function here
    else:
        raise Exception("No such mode: " + mode)

    if checkpoint_path is not None:
        model, optimizer, epoch, train_loss = load_checkpoint(checkpoint_path, model, optimizer)
        print('load checkpoint from: ', checkpoint_path)
        print('epoch: ', epoch)
        print('train_loss: ', train_loss)
        print('model: ', model)
        print('optimizer: ', optimizer)

    wandb.watch(model)
    for epoch in range(total_epochs):
        print('Epoch: ', epoch)
        train_loss = train(model, optimizer, loss_fn, train_loader)
        val_loss = evaluate(model, loss_fn, val_loader)
        wandb.log({"train/train_loss": train_loss, "val/val_loss": val_loss, "epoch": epoch})
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}")
        # create a folder named checkpoint if not exist
        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint')
        save_checkpoint(total_epochs, model, optimizer, train_loss, path='./checkpoint/' + mode + '_latest.pth')

    test_loss = test_accuracy(model, loss_fn, test_loader)
    wandb.log({"test/test_loss": test_loss})


if __name__ == "__main__":
    # I used wandb for logging and visualization
    # it helps lots when doing remote monitoring and hyperparameter tuning
    mode = sys.argv[1]
    try:
        checkpoint_path = sys.argv[2]
        display_name = mode + '_resume_from_checkpoint'
    except IndexError:
        checkpoint_path = None
        display_name = mode + '_new'
    wandb.init(project="imagenet1k", name=display_name)
    run(mode, checkpoint_path)

