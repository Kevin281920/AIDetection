import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from Model import *
from Utils import *
from Config import ModelConfig
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.disable()
class Evaluator:
    def __init__(self, checkpointPath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.font = Path("./calibril.ttf")
        checkpoint = torch.load(checkpointPath, map_location=self.device, weights_only=False)
        startEpochs = checkpoint["epoch"] + 1
        self.model = MainModel(ModelConfig)
        biases = []
        notBiases = []
        for paramName, param in self.model.named_parameters():
            if param.requires_grad:
                if paramName.endswith("bias"):
                    biases.append(param)
                else:
                    notBiases.append(param)

        self.model.load_state_dict(checkpoint["model"])


        self.model = self.model.to(ModelConfig.device)
        self.model.eval()
        print(checkpoint['epoch'])
        self.transforms = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        try:
            self.font = ImageFont.truetype(str(self.font), 15)
        except IOError:
            print(f"Font file {self.font} not found, using default font")
            self.font = ImageFont.load_default()
    def detect(self, img, minScore = 0.5, maxOverlap = 0.5, topK = 200, supressedClasses = None):
        originalImage = Image.open(img).convert('RGB')
        TensorImage = self.transforms(originalImage).unsqueeze(0).to(self.device)
        with torch.no_grad():
            PredictedLoc, PredictedScore = self.model(TensorImage)
        detBoxes, detLabels, detScores = self.model.detectObjects(PredictedLoc, PredictedScore, minScore=minScore, maxOverlap=maxOverlap, topK=topK)
        detBoxes = detBoxes[0].cpu()
        detScores = detScores[0].cpu()
        detLabels = [reverse_label_map[l]
                     for l in detLabels[0].cpu().tolist()]
        if detLabels == ["background"] or len(detLabels) == 0:
            print("No objects found")
            return(originalImage)
        print(len(detBoxes))
        dim = torch.FloatTensor([originalImage.width,originalImage.height, originalImage.width, originalImage.height]).unsqueeze(0)
        detBoxes = detBoxes * dim
        annotatedImage = self._annotate_Image(originalImage, detBoxes, detLabels, detScores, supressedClasses)
        return annotatedImage
    def _annotate_Image(self, image, detBoxes, detLabels, detScores, supressedClasses = None):
        annotatedImage = image.copy()
        draw = ImageDraw.Draw(annotatedImage)
        boxesDrawn = 0
        for i in range(detBoxes.size(0)):
            if supressedClasses and detLabels[i] in supressedClasses:
                continue
            boxLoc = detBoxes[i].tolist()
            score = detScores[i].item()
            if score < 0.1:
                continue
            color = label_color_map[detLabels[i]]
            self._draw_BoundingBox(draw, boxLoc, color)
            self._draw_Label(draw, boxLoc, detLabels[i], score, color)
            boxesDrawn += 1
        return annotatedImage
    def _draw_BoundingBox(self, draw, boxLoc, color):
        draw.rectangle(xy=boxLoc, outline=color, width=3)
        for o in [1.0, 2.0]:
            offsetBox = [c+o for c in boxLoc]
            draw.rectangle(xy=offsetBox, outline=color, width=1)
    def _draw_Label(self, draw, boxLoc, label, score, color):
        try:
            label_text = f"{label.upper()} {score:.2f}"
            text_size = self.font.getsize(label_text)
        except AttributeError:
            label_text = f"{label.upper()} {score:.2f}"
            text_size = (len(label_text) * 10, 15)
            text_location = [boxLoc[0] + 2., boxLoc[1] - text_size[1]]
            textbox_location = [
                boxLoc[0],
                boxLoc[1] - text_size[1],
                boxLoc[0] + text_size[0] + 4.,
                boxLoc[1]
            ]

            # Draw label background
            draw.rectangle(xy=textbox_location, fill=color)

            # Draw text
            draw.text(xy=text_location, text=label_text, fill='white', font=self.font)
def detectSingleImage(img, checkPointPath= "checkpoint0.pth.tar", outputPath = "./", minConfidence = 0.5):
    detector = Evaluator(checkPointPath)
    result = detector.detect(img)
    if outputPath:
        result.save(outputPath)
    return result
def main():
    imagePath = "./VOC2012/JPEGImages/2007_000452.jpg"
    outputPath = "./outputPath"
    result = detectSingleImage(imagePath, minConfidence = 0.1)
if __name__ == '__main__':
    main()
