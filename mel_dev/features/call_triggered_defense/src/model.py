import mindspore as ms
from mindspore import nn


class CallTriggeredDefenseModel(nn.Cell):
    """
    Simple MLP for fraud probability prediction on tabular features.
    """

    def __init__(self, input_dim: int, hidden_units=(64, 32), dropout_rate: float = 0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_units:
            layers.append(nn.Dense(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=1.0 - dropout_rate))
            in_dim = h

        layers.append(nn.Dense(in_dim, 1))
        layers.append(nn.Sigmoid())  # output: fraud probability in [0, 1]

        self.net = nn.SequentialCell(layers)

    def construct(self, x):
        return self.net(x)
