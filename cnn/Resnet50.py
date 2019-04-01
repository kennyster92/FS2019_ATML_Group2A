#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resnet50 structure
"""

import os
import torch
import torch.nn as nn
import math
from torchvision.datasets import ImageFolder #imageFolder is a data loader
from torchvision.transforms import Resize, ToTensor, Normalize, Compose #Composes several transforms together
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

#  TODO link the folder containing images
train_dir = './.../train'
test_dir = './.../test'
val_dir = './.../valid'



root_dir = train_dir

#  TODO choose the image size
target_size = (224,224) #size of the image after transformation
transforms = Compose([Resize(target_size), # Resize image
                    ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                    Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5)), # scales to [-1.0, 1.0]
                    ])

train_dataset = ImageFolder(root_dir, transform=transforms) # takes in an PIL image and returns a transformed version.
# len(train_dataset) #contain all images of the set, dereferencable with train_dataset[x][0] -->23000 
# type(train_dataset)


#  TODO selection batch size
batch_size = 4

# Parameter description --------------------------
#  DataLoader(Dataset,int,bool,int)
#  dataset (Dataset) – dataset from which to load the data.
#  batch_size (int, optional) – how many samples per batch to load (default: 1)
#  shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
#  num_workers = n - how many threads in background for efficient loading

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 

val_root_dir = val_dir
val_dataset = ImageFolder(val_root_dir, transform=transforms)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) #  shuffle true or false?
len(val_dataset) 

#  TODO test if dataset are good implemented, with a label for each image
#  try to iterate over the train dataset
for image, label in train_dataloader:
    print(image.size(), label.size())
    print(label)
    break #  break here just to show 1 batch of data


#  try to iterate over the validation dataset
for image, label in val_dataloader:
    print(image.size(), label.size())
    print(label)
    break #  break here just to show 1 batch of data


#  Resnet_50 implementation---------------------------------------------------------------------------------------------

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs): #  Constructs a ResNet-50 model
    """.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet, we won´t use it here
        bottleneck is a 3 operation block
        [3, 4, 6, 3] are the layers
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)  #  description of the basic structure of resnet

    return model



#  -------------------------------------------------------------------------------------------------------------------


model = resnet50()  # model to use
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9) # learning step and momentum accelerate gradients vectors in the right directions
    



def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function):
    # monitor loss functions as the training progresses
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        # Training phase
         
        correct_train_predictions = 0 # We will measure accuracy
        # Iterate mini batches over training dataset
        losses = []
# --------------------------------------------
# Test phase

        for images, labels in train_dataloader:
            images = images.to(device) #we have to send the inputs and targets at every step to the GPU too
            labels = labels.to(device)
            output = model(images)  #run prediction; output <- vector with probabilities of each class
            # set gradients to zero
            optimizer.zero_grad() 
            loss = loss_function(output, labels)
#             print(loss.item())
            loss.backward()  # computes dloss/dx for every parameter x
            optimizer.step()  # performs a parameter update based on the current gradient

            # Metrics
            losses.append(loss.item()) # gets the a scalar value held in the loss.
            predicted_labels = output.argmax(dim=1)
#            print(predicted_labels)
            n_correct = (predicted_labels == labels).sum().item() #compare the computation with ground truth
            correct_train_predictions += n_correct
        train_losses.append(np.mean(np.array(losses))) #build a losses array on the way
        train_accuracies.append(100.0*correct_train_predictions/len(train_dataloader.dataset)) #ratio of correct answer on ground truth
        
# --------------------------------------------
# Evaluation phase
        
        correct_val_predictions = 0 # We will measure accuracy
        # Iterate mini batches over validation set
        # We don't need gradients
        losses = []
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = loss_function(output, labels)

                losses.append(loss.item())
                predicted_labels = output.argmax(dim=1)
                n_correct = (predicted_labels == labels).sum().item()
                correct_val_predictions += n_correct
        val_losses.append(np.mean(np.array(losses)))
        val_accuracies.append(100.0*correct_val_predictions/len(val_dataloader.dataset))

        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                      train_losses[-1],
                                                                                                      train_accuracies[-1],
                                                                                                      val_losses[-1],
                                                                                                      val_accuracies[-1]))
    return train_losses, val_losses, train_accuracies, val_accuracies



n_epochs = 40
train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_dataloader, val_dataloader, optimizer, n_epochs, criterion)
