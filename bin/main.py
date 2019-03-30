import torch
from argparse import ArgumentParser
import utils.ConfigurationFileParser as conf


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


    model = conf.getModel(config)

    #optimizer = conf.getOptimizer(config)
    #loss_function = conf.getLoss(config)
    #scheduler = conf.getScheduler(config)

    print(model)

    #model = ConvNet()
    #model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
    #loss_function = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    #train_losses, val_losses, train_accuracies, val_accuracies = fit(train_dataset, val_dataset, model, optimizer, loss_function, n_epochs, scheduler)