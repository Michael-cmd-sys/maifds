import mindspore.nn as nn

class ClickTxLinkModel(nn.Cell):
    """
    MLP for Click → Transaction link correlation fraud scoring.
    Outputs LOGITS (no sigmoid). Sigmoid is applied in inference.
    """

    def __init__(self, input_dim: int, hidden_units=(64, 32), dropout_rate: float = 0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_units:
            layers.append(nn.Dense(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))  # ✅ stable API
            in_dim = h

        layers.append(nn.Dense(in_dim, 1))  # ✅ logits only
        self.net = nn.SequentialCell(layers)

    def construct(self, x):
        return self.net(x)
