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

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
# Assign a unique color to each class for visualization
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
def createDataList(VOC07Path, VOC12Path, OutputPath):
    VOC07Path = Path(VOC07Path).absolute()
    VOC12Path = Path(VOC12Path).absolute()
    TrainingImages = []
    TrainingObjects = []
    NumObjects = 0
    OutputPath = Path(OutputPath)
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
    #assert tensor.dim() == m.size()
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
        ourClassImages = ourImages[ourLabels == i]
        ourClassBoxes = ourBoxes[ourLabels == i]
        ourClassScores = ourScores[ourLabels == i]
        classDetections = ourClassScores.size(0)
        if classDetections == 0:
            continue
        ourClassScores, sortIndex = torch.sort(ourClassScores, descending=True, dim=0)
        ourClassImages = ourClassImages[sortIndex]
        ourClassBoxes = ourClassBoxes[sortIndex]
        truePositives = torch.zeros(classDetections, dtype = torch.float).to(device)
        falsePositives = torch.zeros(classDetections, dtype = torch.float).to(device)
        for j in range(classDetections):
            thisBox = ourClassBoxes[j].unsqueeze(0)
            thisImage = ourClassImages[j]
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
                if objectDifficulties[Index] == 0:
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
def GCXGCY_to_CXCY(gcxgcy, priors_cxcy):
    return torch.cat([
        gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
        torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]
    ], dim=1)
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
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if filename is None:
        filename = "checkpoint" + str(epoch) + ".pth.tar"
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")
def expand(image, boxes, filler):
    originalHeight = image.size(1)
    originalWidth = image.size(2)
    maxScale = 4
    scale = random.uniform(1, maxScale)
    newHeight = int(originalHeight * scale)
    newWidth = int(originalWidth * scale)
    filler = torch.FloatTensor(filler)
    newImage = torch.ones((3, newHeight, newWidth), dtype = torch.float) * filler.unsqueeze(1).unsqueeze(1)
    left = random.randint(0, newWidth - originalWidth)
    right = left + originalWidth
    top = random.randint(0, newHeight - originalHeight)
    bottom = top + originalHeight
    newImage[:, top:bottom, left:right] = image
    newBoxes = boxes + torch.FloatTensor([[left, top, left, top]])
    return newImage, newBoxes
def ranCrop(image, boxes, labels, difficulties):
     originalHeight = image.size(1)
     originalWidth = image.size(2)
     while True:
         minOverlap = random.choice([0,0.1,0.3,0.5,0.7,0.9,None])
         if minOverlap is None:
             return image, boxes, labels, difficulties
         maxTrials = 50
         for i in range(maxTrials):
             scaleHeight = random.uniform(0.3,1)
             scaleWidth = random.uniform(0.3,1)
             newHeight = int(originalHeight * scaleHeight)
             newWidth = int(originalWidth * scaleWidth)
             aspectRatio = newHeight / newWidth
             if not 0.5 < aspectRatio < 2:
                 continue
             left = random.randint(0, originalWidth - newWidth)
             right = left + originalWidth
             top = random.randint(0, originalHeight - newHeight)
             bottom = top + originalHeight
             cropped = torch.FloatTensor([left, top, right, bottom])
             overlap = findJaccardOverlap(cropped.unsqueeze(0), boxes)
             overlap = overlap.squeeze(0)
             if overlap.max().item() < minOverlap:
                 continue
             newImage = image[:, top:bottom, left:right]
             boxCenters = (boxes[:, :2] + boxes[:, 2:]) / 2
             croppedCenters = (boxCenters[:, 0] > left) * (boxCenters[:, 0]< right) * (boxCenters[:, 1] > top) * (boxCenters[:, 1] < bottom)
             if not croppedCenters.any():
                 continue
             newBoxes = boxes[croppedCenters,:]
             newLabels = labels[croppedCenters]
             newDifficulties = difficulties[croppedCenters]
             newBoxes = newBoxes.float()
             newBoxes[:, :2] = torch.max(newBoxes[:, :2], cropped[:2])
             newBoxes[:, :2] -= cropped[:2]
             newBoxes[:, 2:] = torch.min(newBoxes[:, 2:], cropped[2:])
             newBoxes[:, 2:] -= cropped[2:]
             return newImage, newBoxes, newLabels, newDifficulties
def flip(image, boxes):
     newImage = TF.hflip(image)
     newBoxes = boxes.clone().float()
     newBoxes[:, 0] = image.width - boxes[:, 0] - 1
     newBoxes[:, 2] = image.width - boxes[:, 2] - 1
     newBoxes = newBoxes[:, [2, 1, 0, 3]]
     return newImage, newBoxes

def resize(image, boxes, dims = (300, 300), returnPercentCords = True):
    newImage = TF.resize(image, dims)
    oldDims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    newBoxes = boxes.float() / oldDims
    if not returnPercentCords:
        newDims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        newBoxes = newBoxes * newDims
    return newImage, newBoxes
def distortions(image):
    newImage = image
    distortions = [TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue]
    random.shuffle(distortions)
    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == "adjust_hue":
                adjustFactor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                adjustFactor = random.uniform(0.5, 1.5)
            newImage = d(newImage, adjustFactor)
    return newImage
def transform(image, boxes, labels, difficulties, split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    newImage = image
    newBoxes = boxes.float() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float)
    newLabels = labels
    newDifficulties = difficulties
    if split == "train":
        newImage = distortions(newImage)
        newImage = TF.to_tensor(newImage)
        if random.random() < 0.5:
            newImage, newBoxes = expand(newImage, newBoxes, filler=mean)
        newImage, newBoxes, newLabels, newDifficulties = ranCrop(newImage, newBoxes, newLabels, newDifficulties)
        newImage = TF.to_pil_image(newImage)
        if random.random() < 0.5:
            newImage, newBoxes = flip(newImage, newBoxes)
    newImage, newBoxes = resize(newImage, newBoxes)
    newImage = TF.to_tensor(newImage)
    newimage = TF.normalize(newImage, mean=mean, std=std)
    return newImage, newBoxes, newLabels, newDifficulties
def accuracy(scores, targets, k):
    batchSize = targets.size(0)
    _, index = scores.topk(k, 1, True, True)
    correct = index.eq(targets.view(-1, 1).expand_as(index))
    correctTotal = correct.view(-1).float().sum().item()
    return correctTotal / (batchSize * 100)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
if __name__ == "__main__":
    createDataList("./VOC2007", "./VOC2012", "./output")
