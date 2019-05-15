import torch


class Tester:

    def __init__(self):
        """
        Initializes a new instance of the Tester class.
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def predict(self, test_dataloader, model):
        '''
        Predicts the labels on a test dataset
        '''

        model.eval()
        n_correct = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.float()
                labels = labels.long()
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images)
                n_correct += torch.sum(output.argmax(1) == labels).item()

        accuracy = 100.0 * n_correct / len(test_dataloader.dataset)
        
        print('Test_accuracy: {:.4f}' .format(accuracy))

        return accuracy

