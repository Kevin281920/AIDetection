from dataclasses import dataclass
from pathlib import Path
import torch
@dataclass
class trainingConfig:
    batch_size = 64
    learning_rate = 1e-3
    epochs = 50
    weight_decay = 5e-4
    momentum = 0.9
    decay_LR = (8000, 10000)
    gradient_clipping = None
@dataclass
class dataConfig:
    dataFolder = Path("./")
    keepDifficult = True
    workers = 4
class modelConfig:
    num_classes = 21
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
