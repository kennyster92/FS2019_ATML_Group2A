import json


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
        self.channels = int(values['config']['n_channels'])
        self.lr = float(values['config']['learning_rate'])
        self.dropout = float(values['config']['dropout_p'])

        jsonFile.close()

    def getModel(self):
        if self.model == 'unet2d':
            return 'unet2d'
        elif True:
            return None
        else:
            return None

    def getOptimizer(self):
        if self.optimizer == 'SGD':
            return 'SGD'
        elif True:
            return None
        else:
            return None

    def getLoss(self):
        if self.loss == 'CrossEntropy':
            return 'CrossEntropy'
        elif True:
            return None
        else:
            return None

    def getScheduler(self):
        if self.scheduler == 'stepLR':
            return 'stepLR'
        elif True:
            return None
        else:
            return None

    def getEpochs(self):
        return self.epochs

    def getBatchSize(self):
        return self.batchSize

    def getChannels(self):
        return self.channels

    def getLearningRate(self):
        return self.lr

    def getDropout(self):
        return self.dropout

