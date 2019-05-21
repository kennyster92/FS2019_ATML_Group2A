import torch
import numpy as np
import os


class Trainer:

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def fit(self, train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, scheduler=None):
        '''
        Trains the model for all the epochs
        '''

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train(model, train_dataloader, optimizer, loss_fn, epoch)
            val_loss, val_accuracy = self.validate(model, val_dataloader, loss_fn)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            if scheduler:
                scheduler.step()  # argument only needed for ReduceLROnPlateau
            print(
                'Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
                    epoch + 1, n_epochs,
                    train_losses[-1],
                    train_accuracies[-1],
                    val_losses[-1],
                    val_accuracies[-1]))

        return train_losses, train_accuracies, val_losses, val_accuracies

    def train(self, model, train_loader, optimizer, loss_fn, epoch):
        '''
        Trains the model for one epoch
        '''

        model.train()
        losses = []
        n_correct = 0
        for iteration, (images, labels) in enumerate(train_loader):
            images = images.float()
            labels = labels.long()
            images = images.to(self.device)
            labels = labels.to(self.device)
            output = model(images)
            optimizer.zero_grad()

            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            n_correct += torch.sum(output.argmax(1) == labels).item()
        accuracy = 100.0 * n_correct / len(train_loader.dataset)

        # Save model to file
        if not os.path.exists("modelTest"):
            os.mkdir("modelTest")

        try:
            torch.save(model.state_dict(), '{}model_{}_{}.pth'.format("modelTest/", '001', epoch))
        except:
            print("Problem during saving model")
        else:
            print("Model saved")

        return np.mean(np.array(losses)), accuracy

    def validate(self, model, validation_loader, loss_fn):
        '''
        Validates the model on data from validation_loader
        '''

        model.eval()
        test_loss = 0
        n_correct = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.float()
                labels = labels.long()
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images)
                loss = loss_fn(output, labels)
                test_loss += loss.item()
                n_correct += torch.sum(output.argmax(1) == labels).item()

        average_loss = test_loss / len(validation_loader)
        accuracy = 100.0 * n_correct / len(validation_loader.dataset)
        torch.save(model.state_dict(), '{}model_{}_afterValSave.pth'.format("modelTest/", '001'))
        return average_loss, accuracy



