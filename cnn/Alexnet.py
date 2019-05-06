#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alexnet structure
"""


import torch

from torchvision.transforms import Resize, ToTensor, Normalize, Compose #Composes several transforms together
from torch.utils.data import DataLoader
import numpy as np

# TODO: Delete all the unnecessary lines of code... keep only the model class
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()
# print(device)
#
# #  TODO link the folder containing images
# images_file = './data/images.npy'
# labels_file = './data/labels.npy'
#
#
# #root_dir = train_dir
#
# #  TODO choose the image size
# target_size = (224,224) #size of the image after transformation
# transforms = Compose([Resize(target_size), # Resize image
#                     ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
#                     Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5)), # scales to [-1.0, 1.0]
#                     ])
#
# images = np.load(images_file)
# labels = np.load(labels_file)
#
#
# #  TODO selection batch size
# batch_size = 4

# creation of the Dataset class --------------------------


# Parameter description --------------------------
#  DataLoader(Dataset,int,bool,int)
#  dataset (Dataset) – dataset from which to load the data.
#  batch_size (int, optional) – how many samples per batch to load (default: 1)
#  shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
#  num_workers = n - how many threads in background for efficient loading
#
#
# transforms = Compose([ToTensor()])
#
# #  TODO split data in 3 different sent of images
# train_dataset = DigitDataset(train_im, train_label, transforms)
# val_dataset = DigitDataset(val_im, va_label, transforms)
# test_dataset = DigitDataset(test_im, test_label, transforms)
#
# # use dataloader and give dataset in parameter
#
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
#
#
#
#
# #  TODO test if dataset are good implemented, with a label for each image
# #  try to iterate over the train dataset
# for image, label in train_dataloader:
#     print(image.size(), label.size())
#     print(label)
#     break #  break here just to show 1 batch of data
#
#
# #  try to iterate over the validation dataset
# for image, label in val_dataloader:
#     print(image.size(), label.size())
#     print(label)
#     break #  break here just to show 1 batch of data

#  Alexnet implementation---------------------------------------------------------------------------------------------


import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 192, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        print(x.size())
        x = x.view(x.size(0), 256 * 192)
        x = self.classifier(x)
        return x


    def alexnet(pretrained=False, **kwargs):
        r"""AlexNet model architecture from the
        `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = AlexNet(**kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        return model

#  -------------------------------------------------------------------------------------------------------------------

#
# import torch.optim as optim
#
# model = AlexNet()
# model = model.to(device)  # transfer the neural net onto the GPU
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # learning step and momentum accelerate gradients vectors in the right directions
#
#
#
# import numpy as np
#
#
# def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function):
#     # monitor loss functions as the training progresses
#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []
#
#     for epoch in range(n_epochs):
#         # Training phase
#
#         correct_train_predictions = 0  # We will measure accuracy
#         # Iterate mini batches over training dataset
#         losses = []
#         # --------------------------------------------
#         # Test phase
#
#         for images, labels in train_dataloader:
#             images = images.to(device)  # we have to send the inputs and targets at every step to the GPU too
#             labels = labels.to(device)
#             output = model(images)  # run prediction; output <- vector with probabilities of each class
#             # set gradients to zero
#             optimizer.zero_grad()
#             loss = loss_function(output, labels)
#             #             print(loss.item())
#             loss.backward()  # computes dloss/dx for every parameter x
#             optimizer.step()  # performs a parameter update based on the current gradient
#
#             # Metrics
#             losses.append(loss.item())  # gets the a scalar value held in the loss.
#             predicted_labels = output.argmax(dim=1)
#             #            print(predicted_labels)
#             n_correct = (predicted_labels == labels).sum().item()  # compare the computation with ground truth
#             correct_train_predictions += n_correct
#         train_losses.append(np.mean(np.array(losses)))  # build a losses array on the way
#         train_accuracies.append(100.0 * correct_train_predictions / len(
#             train_dataloader.dataset))  # ratio of correct answer on ground truth
#
#         # --------------------------------------------
#         # Evaluation phase
#
#         correct_val_predictions = 0  # We will measure accuracy
#         # Iterate mini batches over validation set
#         # We don't need gradients
#         losses = []
#         with torch.no_grad():
#             for images, labels in val_dataloader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 output = model(images)
#                 loss = loss_function(output, labels)
#
#                 losses.append(loss.item())
#                 predicted_labels = output.argmax(dim=1)
#                 n_correct = (predicted_labels == labels).sum().item()
#                 correct_val_predictions += n_correct
#         val_losses.append(np.mean(np.array(losses)))
#         val_accuracies.append(100.0 * correct_val_predictions / len(val_dataloader.dataset))
#
#         print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
#             epoch + 1, n_epochs,
#             train_losses[-1],
#             train_accuracies[-1],
#             val_losses[-1],
#             val_accuracies[-1]))
#     return train_losses, val_losses, train_accuracies, val_accuracies
#
#
# n_epochs = 20
# train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_dataloader, val_dataloader, optimizer,
#                                                                    n_epochs, criterion)
#