import matplotlib.pyplot as plt
import numpy as np

# file responsible for the plotting of graphs

def plot_loss(n_epochs, train_losses, val_losses, experiment_name):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses)  # display evenly scale with arange
    plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/val loss')
    plt.savefig('Plot_loss_figure_exp_' + experiment_name + '.png')

def plot_accuracy(n_epochs, train_accuracy, val_accuracy, experiment_name):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_accuracy)  # display evenly scale with arange
    plt.plot(np.arange(n_epochs), val_accuracy)
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy value')
    plt.title('Train/val accuracy')
    plt.savefig('Plot_accuracy_figure_exp_' + experiment_name + '.png')
    
    
def plot_Test_loss(n_epochs, test_losses, experiment_name):
    plt.figure()
    plt.plot(np.arange(n_epochs), test_losses)  # display evenly scale with arange
    plt.legend(['test_losses'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Test loss')
    plt.savefig('Plot_Test_loss_figure_exp_' + experiment_name + '.png')

def plot_test_accuracy(n_epochs, test_accuracy, experiment_name):
    plt.figure()
    plt.plot(np.arange(n_epochs), test_accuracy)  # display evenly scale with arange
    plt.legend(['test_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy value')
    plt.title('Test accuracy')
    plt.savefig('Plot_test_accuracy_figure_exp_' + experiment_name + '.png')