import json
import torch.nn as nn
import cnn.Alexnet as alexnet
from cnn import Resnet50 as resnet
import torch.optim as optim


class ConfigurationFileParser:

    def __init__(self, config):
        self.config = config
        jsonFile = open(config, 'r')
        values = json.load(jsonFile)

        # load the necessary parameters to run the learning from the json file
        self.model = values['config']['model']
        self.optimizer = values['config']['optimizer']
        self.loss = values['config']['loss_function']
        self.scheduler = values['config']['scheduler']
        self.epochs = int(values['config']['epochs'])
        self.batchSize = int(values['config']['batch_size'])
        self.lr = float(values['config']['learning_rate'])
        self.model_dir = values['config']['model_dir']
        self.experiment_name = values['config']['experiment_name']
        self.pretrained = values['config']['pretrained']

        jsonFile.close()
        
        
        #Add print parameters used
        print("Parameters read from config files:")
        print("Model:", self.model)
        print("Optimizer:", self.optimizer)
        print("Loss_function:", self.loss)
        print("Number of epochs:", self.epochs)
        print("Batch_size:", self.batchSize)
        print("Learning rate:", self.lr, "\n")

    def getModel(self, pretrained):
        if self.model == 'alexnet':
            return alexnet.AlexNet(pretrained)
        elif self.model == 'resnet50':
            return resnet.resnet50(pretrained)
        else:
            return None

    def getOptimizer(self, model, lr):
        if self.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif self.optimizer == 'adam':
            return optim.Adam(model.parameters(), lr=lr, eps=1e-08, weight_decay=0.04, amsgrad=False)
        else:
            return None

    def getLoss(self):
        if self.loss == 'CrossEntropy':
            return nn.CrossEntropyLoss()
        else:
            return None

    def getScheduler(self, optimizer):
        if self.scheduler == 'stepLR':
            return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
        elif self.scheduler == 'none':
            return None
        else:
            return None

    def getEpochs(self):
        return self.epochs

    def getBatchSize(self):
        return self.batchSize

    def getLearningRate(self):
        return self.lr

    def getModelDir(self):
        return self.model_dir

    def getExperimentName(self):
        return str(self.experiment_name)

    def getPretrained(self):
        if self.pretrained == 'true':
            return True
        elif self.pretrained == 'false':
            return False
        else:
            return None

