import mindspore.nn as nn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CKPT_PATH = os.path.join(BASE_DIR, "proactive_warning_mlp.ckpt")
MEAN_PATH = os.path.join(BASE_DIR, "feature_mean.npy")
STD_PATH  = os.path.join(BASE_DIR, "feature_std.npy")



class ProactiveWarningModel(nn.Cell):
    """
    MLP that predicts whether a user should receive a proactive
    scam warning (should_warn = 1).
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

        # final output: probability in [0, 1]
        layers.append(nn.Dense(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.SequentialCell(layers)

    def construct(self, x):
        return self.net(x)
