import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """

    def __init__(self, input_dim, output_dim, hidden_layers=[32, 32], activation=nn.ReLU):
        super().__init__()
        layers = []
        in_d = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(activation())
            in_d = h_dim

        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)