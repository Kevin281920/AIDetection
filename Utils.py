import json
import os
from tokenize import String
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k:v+1 for v,k in enumerate(voc_labels)}
label_map["Background"] = 0
reverse_label_map = {v:k for k,v in label_map.items()}
def createDataList(VOC07Path, VOC12Path, OutputPath):
    VOC07Path = os.path.abspath(VOC07Path)
    VOC12Path = os.path.abspath(VOC12Path)
    TrainingImages = []
    TrainingObjects = []
    NumObjects = 0
    for path in [VOC07Path, VOC12Path]:
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt' )) as f:
            ids = f.read().splitlines()
            print("Found Files")



if __name__ == "__main__":
    createDataList("./VOC2007", "./VOC2012", "./output")