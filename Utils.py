import json
from pathlib import Path
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
    VOC07Path = Path(VOC07Path).absolute()
    VOC12Path = Path(VOC12Path).absolute()
    TrainingImages = []
    TrainingObjects = []
    NumObjects = 0
    for path in [VOC07Path, VOC12Path]:
        with open(path/"ImageSets/Main/trainval.txt") as f:
            ids = f.read().splitlines()
            print("Found Files")
        for id in ids:
            annotationPath = path/"Annotations"/f"{id}.xml"
            if not annotationPath.exists():
                continue
            objects = parseAnnotations(annotationPath)
            if len(objects["boxes"]) == 0:
                continue
            NumObjects += len(objects["boxes"])
            TrainingObjects.append(objects)
            imagePath = path/"JPEGImages"/f"{id}.jpg"
            if imagePath.exists():
                TrainingImages.append(str(imagePath))
                print("Parsed XML files")
    assert len(TrainingObjects)==len(TrainingImages)
    with open(OutputPath/"trainimages.json", "w") as f:
        json.dump(TrainingImages, f)
    with open(OutputPath / "trainobjects.json", "w") as f:
        json.dump(TrainingObjects, f)
    with open(OutputPath/"labelmap.json", "w") as f:
        json.dump(label_map, f)
        print("Training data saved to JSON files")
    TrainingImages = []
    TrainingObjects = []
    NumObjects = 0
    with open(VOC12Path/"ImageSets/Main/test.txt") as f:
        ids = f.read().splitlines()
        print("Found Test Images ID's")
    for id in ids:
        annotationPath = VOC12Path/"Annotations"/f"{id}.xml"
        if not annotationPath.exists():
            continue
        objects = parseAnnotations(annotationPath)
        if len(objects["boxes"]) == 0:
            continue
        NumObjects += len(objects["boxes"])
        TrainingObjects.append(objects)
        imagePath = VOC12Path/"JPEGImages"/f"{id}.jpg"
        if imagePath.exists():
            TrainingImages.append(str(imagePath))
            print("Parsed XML files")
    print("Parsed Test Images ID's")
    assert len(TrainingObjects)==len(TrainingImages)
    with open(OutputPath / "testimages.json", "w") as f:
        json.dump(TrainingImages, f)
    with open(OutputPath / "testobjects.json", "w") as f:
        json.dump(TrainingObjects, f)
        print("Test files saved to JSON files")
def parseAnnotations(path):
    boxes = [ ]
    labels = [ ]
    difficulty = [ ]
    tree = ET.parse(path)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        label = obj.find('name').text.lower().strip()
        if label not in label_map:
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) -1
        ymin = int(bbox.find('ymin').text) -1
        xmax = int(bbox.find('xmax').text) -1
        ymax = int(bbox.find('ymax').text) -1
        difficulty.append(difficult)
        labels.append(label_map[label])
        boxes.append([xmin, ymin, xmax, ymax])
    return {"boxes": boxes, "labels": labels, "difficulty": difficulty}
def Decimate(tensor, m):
    assert tensor.dim() == m.size()
    for d in range(tensor.dim()):
        if m[d] != None:
            indices = torch.arange(start=0, end=tensor.size(d), step=m[d])
            tensor = tensor.index_select(d, indices.long())
    return tensor
def calcmap(ourBoxes, ourLabels, ourScores, trueBoxes, trueLabels, trueScores):
    assert ourBoxes.size() == ourScores.size() == ourLabels.size()
    numClasses = len(label_map)
    trueImages = []
    for i in range(len(trueLabels)):
        trueImages.extend([i]*trueLabels[i].size(0))
    trueImages = torch.longtensor(trueImages).to(device)
    trueBoxes = torch.cat(trueBoxes, dim=0)
    trueLabels = torch.cat(trueLabels, dim=0)
    trueScores = torch.cat(trueScores, dim=0)
    assert trueImages.size() == trueScores.size() == trueBoxes.size()
    ourImages = []
    for i in range(len(ourLabels)):
        ourImages.extend([i]*ourLabels[i].size(0))
    ourImages = torch.longtensor(ourImages).to(device)
    ourBoxes = torch.cat(ourBoxes, dim=0)
    ourLabels = torch.cat(ourLabels, dim=0)
    ourScores = torch.cat(ourScores, dim=0)
    assert ourImages.size() == ourScores.size() == ourBoxes.size()
    averagePrecision = torch.zeros(numClasses - 1, dtype=torch.float)
    for i in range(1, numClasses):
        trueClassImage = trueImages[trueLabels == i]
        trueClassBoxes = trueBoxes[trueLabels == i]
        trueClassScores = trueScores[trueLabels == i]
        nEasyClass = (1-trueClassScores).sum().item()
        trueClassBoxesDetected = torch.zeros(trueClassScores.size(0), dtype = torch.bool).to(device)
        ourClassImages = trueImages[trueLabels == i]
        ourClassBoxes = trueBoxes[trueLabels == i]
        ourClassScores = trueScores[trueLabels == i]
        classDetections = ourClassScores.size(0)
        if classDetections == 0:
            continue
        ourClassScores, sortIndex = torch.sort(ourClassScores, descending=True, dim=0)
        ourClassImages = ourClassImages[sortIndex]
        ourClassBoxes = ourClassBoxes[sortIndex]
        truePositives = torch.zeros(classDetections, dtype = torch.float).to(device)
        falsePositives = torch.zeros(classDetections, dtype = torch.float).to(device)

if __name__ == "__main__":
    createDataList("./VOC2007", "./VOC2012", "./output")
