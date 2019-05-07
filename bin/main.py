from itertools import dropwhile

import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from argparse import ArgumentParser
from utils import ConfigurationFileParser as conf
from bin.test import Testing as test
from bin.train import Training as train
from bin.train import MelanomaDataset as data
import matplotlib.pyplot as plt

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
    print(device)

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
    model_dir = configuration.getModelDir()

    # TODO: read images and labels from files
    images_file = '../data/BenignAndMalignant20000DatasetIMG.npy'
    labels_file = '../data/BenignAndMalignant20000DatasetTAG.npy'

    images = np.load(images_file)
    labels = np.load(labels_file)

    train_imgs = np.concatenate([images[0:9000], images[10000:19000]]) #  18000
    train_labels = np.concatenate([labels[0:9000], labels[10000:19000]])

    val_imgs =  np.concatenate([images[9000:9500], images[19000:19500]])  # 1000
    val_labels =  np.concatenate([labels[9000:9500], labels[19000:19500]])

    test_imgs = np.concatenate([images[9500:10000], images[19500:20000]])  # 1000
    test_labels = np.concatenate([labels[9500:10000], labels[19500:20000]])


    Transforms = Compose([ToTensor()])
    train_dataset = data.MelanomaDataset(train_imgs, train_labels, transform=Transforms)
    val_dataset = data.MelanomaDataset(val_imgs, val_labels, transform=Transforms)
    test_dataset = data.MelanomaDataset(test_imgs, test_labels, transform=Transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    # for image, label in train_dataloader:
    #     print(image.size(), label.size())
    #     selected_im = 2
    #     print(label[selected_im])
    #     im2show = image[selected_im]
    #     print(im2show.size())


    # Train of the model
    trainer = train.Trainer()

    # model = lm.LinearModel(147456)

    # model = alexnet.AlexNet()
    # model = model.to(device)  # transfer the neural net onto the GPU
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    model = resnet.resnet50()
    model = model.to(device)  # transfer the neural net onto the GPU
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # for ResNet


    train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(train_dataloader, val_dataloader, model, optimizer, loss, epochs)


    def plot_loss(n_epochs, train_losses, val_losses):
        plt.figure()
        plt.plot(np.arange(n_epochs), train_losses)  # display evenly scale with arange
        plt.plot(np.arange(n_epochs), val_losses)
        plt.legend(['train_loss', 'val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.title('Train/val loss')


    plot_loss(epochs, train_losses, val_losses)


    def plot_acc(n_epochs, train_accuracy, val_accuracy):
        plt.figure()
        plt.plot(np.arange(n_epochs), train_accuracy)  # display evenly scale with arange
        plt.plot(np.arange(n_epochs), val_accuracy)
        plt.legend(['train_accuracy', 'val_accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy value')
        plt.title('Train/val accuracy');
    plot_acc(epochs, train_accuracy, val_accuracy )


    # save model to file
    # torch.save(model.state_dict(), model_dir)


    # load model from a file
    #model = alexnet.AlexNet(*args, **kwargs)
    #model.load_state_dict(torch.load(model_dir))
    #model.eval()

    # Test of the model
    # tester = test.Tester()
    # test_losses, test_accuracies = tester.predict(test_dataloader, model, optimizer, loss)
