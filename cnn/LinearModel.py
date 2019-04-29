import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2, bias=True)

    def forward(self, input):
        out = input.view(input.size(0), -1)
        out = self.fc(out)

        return out