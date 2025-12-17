from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes
from Utils import transform
from Config import DataConfig
class Data (Dataset):
    def __init__(self, DataConfig, Split):
        self.Split = Split.upper()
        self.DataConfig = DataConfig
        self._validatesplit()
        data = self._loadData()
        if data is None:
            raise RuntimeError("Failed to load data")
        self.images, self.objects = data
    def _validatesplit(self):
        if self.Split not in ["TRAIN", "TEST"]:
            raise ValueError("Split must be TRAIN or TEST")
    def _loadData(self):
        imagePath = self.DataConfig.dataFolder/f"{self.Split}images.json"
        objectPath = self.DataConfig.dataFolder/f"{self.Split}objects.json"
        if not imagePath.exists():
            return None
        if not objectPath.exists():
            return None
        try:
            with imagePath.open("r") as f:
                images = json.load(f)
            with objectPath.open("r") as f:
                objects = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load images and objects from {self.DataConfig.imagePath}{e}")
            return None
        if len(images) != len(objects):
            return None
        return images, objects
    def __getItem__(self, index):
        imagePath = Path(self.images[index])
        if not imagePath.exists():
            altPath = self.DataConfig.imagePath/Path(self.images[index]).name
            if altPath.exists():
                imagePath = altPath
            else:
                raise FileNotFoundError(f"Failed to find {altPath}")
        image = Image.open(imagePath).convert("RGB")
        objects = self.objects[index]
        boxes = BoundingBoxes(objects["boxes"], format="XYXY", canvas_size=image.size[::-1])
        labels = torch.tensor(objects["labels"], dtype=torch.long)
        difficulties = torch.tensor(objects["difficulties"], dtype=torch.bool)
        if not self.DataConfig.keepDifficult:
            keepMask = ~difficulties
            boxes = boxes[keepMask]
            labels = labels[keepMask]
            difficulties = difficulties[keepMask]
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.Split)
        return image, boxes, labels, difficulties
    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        image, boxes, labels, difficulties = zip(*batch)
        return (torch.stack(image, dim=0),list(boxes),list(labels),list(difficulties))

