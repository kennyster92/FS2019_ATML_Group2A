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
    parser.add_argument("-q", "--quiet",
                        dest="verbose", default=True,
                        help="Don't print status messages to stdout")

    args = parser.parse_args()

    config = args.__getattribute__('config_file_path')
    verbose = args.__getattribute__('verbose')

    configuration = conf.ConfigurationFileParser(config)

    model = configuration.getModel()
    optimizer = configuration.getOptimizer()
    loss = configuration.getLoss()
    scheduler = configuration.getScheduler()
    epochs = configuration.getEpochs()
    batchSize = configuration.getBatchSize()
    channels = configuration.getChannels()
    lr = configuration.getLearningRate()
    dropout = configuration.getDropout()

    train_dataloader = None
    val_dataloader = None

    if (verbose):
        print(model)
        print(optimizer)
        print(loss)
        print(scheduler)
        print(epochs)
        print(batchSize)
        print(channels)
        print(lr)
        print(dropout)


    trainer = train.Trainer()
    train_losses, train_accuracies, val_losses, val_accuracies = trainer.fit(train_dataloader, val_dataloader, model, optimizer, loss, epochs, scheduler)
