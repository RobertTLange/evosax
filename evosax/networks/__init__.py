from .mlp import MLP, TanhMLP, DiscreteMLP, ContinuousMLP
from .cnn import CNN, All_CNN_C
from .lstm import LSTM, TanhLSTM, DiscreteLSTM, ContinuousLSTM


# Helper that returns model based on string name
NetworkMapper = {
    "MLP": MLP,
    "TanhMLP": TanhMLP,
    "DiscreteMLP": DiscreteMLP,
    "ContinuousMLP": ContinuousMLP,
    "CNN": CNN,
    "All_CNN_C": All_CNN_C,
    "LSTM": LSTM,
    "TanhLSTM": TanhLSTM,
    "DiscreteLSTM": DiscreteLSTM,
    "ContinuousLSTM": ContinuousLSTM,
}

__all__ = [
    "MLP",
    "TanhMLP",
    "DiscreteMLP",
    "ContinuousMLP",
    "All_CNN_C",
    "LSTM",
    "TanhLSTM",
    "DiscreteLSTM",
    "ContinuousLSTM",
    "NetworkMapper",
]
