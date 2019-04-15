from itertools import dropwhile

import torch
from argparse import ArgumentParser
import utils.ConfigurationFileParser as conf
import bin.test.Testing as test
import bin.train.Training as train

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

    configuration = conf.ConfigurationFileParser(config)

    model = configuration.getModel()
    optimizer = configuration.getOptimizer()
    loss = configuration.getLoss()
    scheduler = configuration.getScheduler()
    epochs = configuration.getEpochs()
    batch_size = configuration.getBatchSize()
    channels = configuration.getChannels()
    lr = configuration.getLearningRate()
    dropout = configuration.getDropout()

    # TODO: implement Dataset class
    train_dataset = Dataset(data_path + "/train")
    val_dataset = Dataset(data_path + "/val")
    test_dataset = Dataset(data_path + "/test")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if (verbose):
        print(model)
        print(optimizer)
        print(loss)
        print(scheduler)
        print(epochs)
        print(batch_size)
        print(channels)
        print(lr)
        print(dropout)

    # Train of the model
    trainer = train.Trainer()
    train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(train_dataloader, val_dataloader, model, optimizer, loss, epochs, scheduler)


    # Test of the model
    # tester = test.Tester()
    # test_losses, test_accuracies = tester.predict(test_dataloader, model, optimizer, loss)
