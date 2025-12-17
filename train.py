import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm
from Model import *
from DatasetImport import *
from Utils import *
from Config import *
from Dataset import *
def main():
    dataConfig = DataConfig()
    modelConfig = ModelConfig()
    trainConfig = TrainingConfig()
    cudnn.benchmark = True
    checkPointPath = trainConfig.checkpoint
    if checkPointPath is None:
        print("Initializing New Model")
        startEpochs = 0
        model = MainModel(modelConfig)
        biases = []
        notBiases = []
        for paramName, param in model.named_parameters():
            if param.requires_grad:
                if paramName.endswith("bias"):
                    biases.append(param)
                else: notBiases.append(param)
        optimizer = torch.optim.SGD(params=[{"params": biases, "lr": 2 * trainConfig.learning_rate}, {'params': notBiases}], lr=trainConfig.learning_rate, momentum=trainConfig.momentum, weight_decay=trainConfig.weight_decay)
    else:
        print("Loading From Checkpoint")
        checkpoint = torch.load(checkPointPath)
        startEpochs = checkpoint["epochs"] + 1
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
    model = model.to(modelConfig.device)
    criterion = multiboxloss(priorscxcy=model.priorscxcy)
    criterion = model.to(modelConfig.device)
    trainDataSet = Data(dataConfig, Split="train")
    trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=trainConfig.batch_size, shuffle=True, collate_fn=Data.collate_fn, num_workers=dataConfig.workers, pin_memory=True)
    iterPerEpoch = len(trainDataSet) // trainConfig.batch_size
    epochs = trainConfig.iter // (iterPerEpoch if iterPerEpoch > 0 else 1)
    decayLRAtEpoch = [iteration // (iterPerEpoch if iterPerEpoch > 0 else 1) for iteration in trainConfig.decay_LR]
    for epoch in range(startEpochs, epochs):
        if epoch in decayLRAtEpoch:
            adjustLearningRate(optimizer, trainConfig.decayed_LR)
        trainMetrics = train(trainLoader = trainLoader, model = model, criterion = criterion, optimizer = optimizer, epoch = epoch, device = modelConfig.device, grad_Clip = trainConfig.gradient_clipping, printfrec = 10)
        print(f"Epoch: {epoch} | Train Loss: {trainMetrics["loss"]:.4f}")
        if epoch % 10 == 0 or epoch == epochs - 1:
            saveCheckPoint(epoch, model, optimizer)
            print(f"Checkpoint Epoch: {epoch}")

def train(trainLoader, model, criterion, optimizer, epoch, device, grad_Clip = None, printfrec = 10):
    model.train()
    BatchTime = AverageMeter()
    dataTime = AverageMeter()
    lossTime = AverageMeter()
    startTime = time.time()
    for i, (image, boxes, labels, difficulties) in enumerate(tqdm(trainLoader, desc=f"Epoch: {epoch}")):
        dataTime.update(time.time() - startTime)
        images = image.to(device,non_blocking=True)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        predictedLocs, predictedScores = model(images)
        loss = criterion(predictedLocs,predictedScores,boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        if grad_Clip is not None:
            clipGradient(optimizer, grad_Clip)
        optimizer.step()
        lossTime.update(loss.item(), image.size(0))
        BatchTime.update(time.time() - startTime)
        startTime = time.time()
        if i % printfrec == 0:
            print(f"Epoch: {epoch}{i}/{len(trainLoader)} | Loss: {lossTime.val:.4f}{lossTime.avg:.4f}")
    return {"loss": lossTime.avg, "BatchTime": BatchTime.avg, "dataTime": dataTime.avg}
if __name__ == "__main__":
    main()