from itertools import dropwhile

import torch
import numpy as np
from argparse import ArgumentParser
from utils import ConfigurationFileParser as conf
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

    # Parameters that can be given by command line

    parser = ArgumentParser(description='Deep learning for melanoma detection')
    parser.add_argument(
        '--config_file',
        dest="config_file_path",
        type=str,
        default='./config/config.json',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--data_path',
        dest="data_path",
        type=str,
        default='../data',
        help='Path to the data.'
    )
    parser.add_argument("-q", "--quiet",
                        dest="verbose", default=True,
                        help="Don't print status messages to stdout")

    args = parser.parse_args()

    config = args.__getattribute__('config_file_path')
    data_path = args.__getattribute__('data_path')
    verbose = args.__getattribute__('verbose')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configuration = conf.ConfigurationFileParser(config)

    model = configuration.getModel()
    #optimizer = configuration.getOptimizer()
    loss = configuration.getLoss()
    scheduler = configuration.getScheduler()
    epochs = configuration.getEpochs()
    batch_size = configuration.getBatchSize()
    channels = configuration.getChannels()
    lr = configuration.getLearningRate()
    dropout = configuration.getDropout()

    # TODO: read images and labels from files
    images_file = '../data/BenignAndMalignant20000DatasetIMG.npy'
    labels_file = '../data/BenignAndMalignant20000DatasetTAG.npy'

    images = np.load(images_file)
    labels = np.load(labels_file)

    train_imgs = images[0:18000]
    train_labels = labels[0:18000]
    val_imgs = images[18000:19000]
    val_labels = labels[18000:19000]
    test_imgs = images[19000:20000]
    test_labels = labels[19000:20000]

    # TODO: shuffle the arrays, now there are before only negative cases and then positive cases

    train_dataset = data.MelanomaDataset(train_imgs, train_labels)
    val_dataset = data.MelanomaDataset(val_imgs, val_labels)
    test_dataset = data.MelanomaDataset(test_imgs, test_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Train of the model
    trainer = train.Trainer()

    # model = lm.LinearModel(147456)

    model = alexnet.AlexNet()
    model = model.to(device)  # transfer the neural net onto the GPU
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    #model = resnet.resnet50()
    #model = model.to(device)  # transfer the neural net onto the GPU
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)  # for ResNet


    train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(train_dataloader, val_dataloader, model, optimizer, loss, epochs)


    # Test of the model
    # tester = test.Tester()
    # test_losses, test_accuracies = tester.predict(test_dataloader, model, optimizer, loss)
