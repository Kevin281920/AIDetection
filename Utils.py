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
        for j in range(classDetections):
            thisBox = ourBoxes[j].unsqueeze(0)
            thisImage = ourImages[j]
            objectBoxes = trueClassBoxes[trueClassImage == thisImage]
            objectDifficulties = trueClassScores[trueClassImage == thisImage]
            if objectBoxes.size(0) == 0:
                falsePositives[j] = 1
                continue
            overlaps = findJaccardOverlap(thisBox, objectBoxes)
            maxOverlap, Index = torch.max(overlaps.squeeze(0), 0)
            originalIndex = torch.LongTensor(range(trueClassBoxes.size(0)))[trueClassImage == thisImage][Index]
            #Confidence Threshold
            if maxOverlap.item() > 0.3:
               if not trueClassBoxesDetected[originalIndex]:
                   truePositives[j] = 1
                   trueClassBoxesDetected[originalIndex] = True
               else:
                   falsePositives[j] = 1
            else:
                falsePositives[j] = 1

            totalTruePositives = torch.cumsum(truePositives, dim=0)
            totalFalsePositives = torch.cumsum(falsePositives, dim=0)
            totalPrecision = totalTruePositives / (totalFalsePositives + totalTruePositives)
            totalObjects = totalTruePositives / nEasyClass
            recallThreshold = torch.arange(0, 1.1, 0.1).tolist()
            Precision = torch.zeros(len(recallThreshold), dtype=torch.float).to(device)
            for r, t in enumerate(recallThreshold):
                aboveT = totalObjects[r] >= t
                if aboveT.any():
                    Precision[r] = totalPrecision[aboveT].max()
                else:
                    Precision[r] = 0
            averagePrecision[i-1] = Precision.mean()
        meanPrecision = averagePrecision.mean().item()
        averagePrecisions = {reverse_label_map[i+1]:j for i, j in enumerate(averagePrecision.tolist())}
        return averagePrecisions, meanPrecision
def findJaccardOverlap(box1, box2):
    intersection = findIntersection(box1, box2)
    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    return intersection / union

def findIntersection(set1, set2):
    lowerBounds = torch.max(set1[:,:2].unsqueeze(1), set2[:,:2].unsqueeze(0))
    upperBounds = torch.min(set1[:,:2].unsqueeze(1), set2[:,:2].unsqueeze(0))
    dimensions = torch.clamp(upperBounds - lowerBounds, min=0)
    dimensions = dimensions[:,:,0] * dimensions[:,:,1]
    return dimensions
def xy_to_CXCY(xy):
    return torch.cat([(xy[:,2:] + xy[:,:2]) / 2, xy[:,2:] - xy[:,:2]], dim=1)
def CXCY_to_XY(CXCY):
    return torch.cat([(CXCY[:,:2] - CXCY[:,2:]) / 2, CXCY[:,:2] + CXCY[:,2:]], dim=1)
def CXCY_to_GCXGCY(CXCY, GCXGCY):
    return torch.cat([(CXCY[:,:2] - GCXGCY[:,:2]) / (GCXGCY[:,2:] / 10), torch.log(CXCY[:,2:]/GCXGCY[:,2:]) * 5], dim=1)
def GCXGCY_to_CXCY(CXCY, GCXGCY):
    return torch.cat([(CXCY[:,:2] * GCXGCY[:,2:]) / 10 + GCXGCY[:,:2], torch.exp(CXCY[2:]/5) * GCXGCY[:,2:]], dim=1)
def adjustLearningRate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale
        print(f"Learning rate changed to {optimizer.param_groups[1]['lr']}")
def clipGradient(optimizer, gradientClip):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-gradientClip, gradientClip)
def saveCheckPoint(epoch, model, optimizer, filename = None):
    state = {"epoch": epoch, "model": model, "optimizer": optimizer}
    if filename is None:
        filename = "checkpoint" + str(epoch) + ".pth.tar"
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

if __name__ == "__main__":
    createDataList("./VOC2007", "./VOC2012", "./output")
