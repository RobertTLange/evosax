from .mlp import MLP
from .cnn import CNN, All_CNN_C
from .lstm import LSTM


# Helper that returns model based on string name
NetworkMapper = {
    "MLP": MLP,
    "CNN": CNN,
    "All_CNN_C": All_CNN_C,
    "LSTM": LSTM,
}

__all__ = [
    "MLP",
    "CNN",
    "All_CNN_C",
    "LSTM",
    "NetworkMapper",
]
