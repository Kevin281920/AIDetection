import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from Model import *
from Utils import *
from Config import ModelConfig
class eval:
    def __init__(self, checkpointPath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.font = Path("./calibril.ttf")
        checkpoint = torch.load(checkpointPath, map_location=self.device, weights_only=False)
        self.model = checkpoint['model'].to(self.device)
        self.model.eval()
        print(checkpoint['epoch'])
        self.transforms = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
