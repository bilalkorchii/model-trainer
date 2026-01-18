import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        init_type="he"
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))

            if batch_norm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(ACTIVATIONS[activation]())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self.output = nn.Sigmoid()

        self._init_weights(init_type)

    def _init_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "xavier":
                    init.xavier_uniform_(m.weight)
                elif init_type == "he":
                    init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif init_type == "lecun":
                    init.normal_(m.weight, 0, (1 / m.in_features) ** 0.5)
                init.zeros_(m.bias)

    def forward(self, x):
        x = self.model(x)
        return self.output(x)
