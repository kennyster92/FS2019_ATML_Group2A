from itertools import dropwhile

import os
import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from argparse import ArgumentParser
from utils import ConfigurationFileParser as conf
from utils import PlotStatistics as plt_stat

from bin.test import Testing as test
from bin.train import Training as train
from bin.train import MelanomaDataset as data

import torch.nn as nn
import torch.optim as optim
from cnn import Alexnet as alexnet
from cnn import Resnet50 as resnet
import cnn.LinearModel as lm


if __name__ == '__main__':
    """The program's entry point.
    Parse the arguments and run the program.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    print("Welcome! The program is running on " + device.__str__() + ". It may take a while...")

    # Parameters that can be given by command line
    parser = ArgumentParser(description='Deep learning for melanoma detection')
    parser.add_argument(
        '--config_file',
        dest="config_file_path",
        type=str,
        default='./bin/config/config1003.json',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--data_path',
        dest="data_path",
        type=str,
        default='./data/',
        help='Path to the data.'
    )
    parser.add_argument("-q", "--quiet",
                        dest="verbose", default=True,
                        help="Don't print status messages to stdout")

    args = parser.parse_args()

    config = args.__getattribute__('config_file_path')
    data_path = args.__getattribute__('data_path')
    verbose = args.__getattribute__('verbose')

    # Reading of the configuration parameters
    configuration = conf.ConfigurationFileParser(config)

    pretrained = configuration.getPretrained()
    model = configuration.getModel(pretrained)
    model = model.to(device)
    lr = configuration.getLearningRate()
    optimizer = configuration.getOptimizer(model, lr)
    loss = configuration.getLoss()
    scheduler = configuration.getScheduler(optimizer)
    epochs = configuration.getEpochs()
    batch_size = configuration.getBatchSize()
    model_dir = configuration.getModelDir()
    experiment_name = configuration.getExperimentName()

    print("I am using configuration file number " + experiment_name)

    # Creation of the datasets
    images_file = data_path + 'BenignAndMalignant20000DatasetIMG.npy'
    labels_file = data_path + 'BenignAndMalignant20000DatasetTAG.npy'

    images = np.load(images_file)
    labels = np.load(labels_file)

    train_imgs = np.concatenate([images[0:9000], images[10000:19000]]) #  18000
    train_labels = np.concatenate([labels[0:9000], labels[10000:19000]])

    val_imgs = np.concatenate([images[9000:9500], images[19000:19500]])  # 1000
    val_labels = np.concatenate([labels[9000:9500], labels[19000:19500]])

    test_imgs = np.concatenate([images[9500:10000], images[19500:20000]])  # 1000
    test_labels = np.concatenate([labels[9500:10000], labels[19500:20000]])

    Transforms = Compose([ToTensor()])
    train_dataset = data.MelanomaDataset(train_imgs, train_labels, transform=Transforms)
    val_dataset = data.MelanomaDataset(val_imgs, val_labels, transform=Transforms)
    test_dataset = data.MelanomaDataset(test_imgs, test_labels, transform=Transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train of the model
    trainer = train.Trainer()

    train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(train_dataloader, val_dataloader, model, optimizer, loss, epochs)

    plt_stat.plot_loss(epochs, train_losses, val_losses, experiment_name)
    plt_stat.plot_accuracy(epochs, train_accuracies, val_accuracies, experiment_name)


    # Save model to file 
    if not os.path.exists("modelTest"):
        os.mkdir("modelTest")
    
    try:
        torch.save(model.state_dict(), '{}model_{}.pth'.format("modelTest/", experiment_name))    
    except:  
        print("Problem during saving model")
    else:
        print("Model saved")


    # Test of the model
    tester = test.Tester()
    test_accuracies = tester.predict(test_dataloader, model)
    
    plt_stat.plot_test_accuracy(epochs, test_accuracies, experiment_name)
