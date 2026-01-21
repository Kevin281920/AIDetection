from dataclasses import dataclass
from pathlib import Path
import torch
@dataclass
class TrainingConfig:
    batch_size = 64
    learning_rate = 1e-3
    epochs = 50
    weight_decay = 5e-4
    momentum = 0.9
    decay_LR = (8000, 10000)
    decayed_LR = 0.1
    gradient_clipping = None
    checkpoint = "./checkpoint0.pth.tar"
    iter = 120000
@dataclass
class DataConfig:
    dataFolder = Path("./")
    keepDifficult = True
    workers = 4
class ModelConfig:
    num_classes = 21
    device = torch.device("cuda")
