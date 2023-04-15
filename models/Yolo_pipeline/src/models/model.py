import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import pytorch_lightning as pl
from torchvision import transforms

from torchvision import models
from torch.utils.data import DataLoader
from src.multi_models.resnet.dataset import FER2013

class EmotionClassifier(pl.LightningModule):
    def __init__(self, model, processed_fer2013, batch_size=128) -> None:
        super(EmotionClassifier, self).__init__()
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.processed_fer2013 = processed_fer2013
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.Normalize(mean=0, std=255)
        ])
        self.test_transform = transforms.Compose([
            transforms.Normalize(mean=0, std = 255)
        ])
        self.batch_size = batch_size
        
    def train_dataloader(self):
        train_set = FER2013(self.processed_fer2013['x_train'], self.processed_fer2013['y_train'], transform=self.train_transform)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=16)
        return train_loader

    def val_dataloader(self):
        val_set = FER2013(self.processed_fer2013['x_val'], self.processed_fer2013['y_val'], transform=self.test_transform)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=16)
        return val_loader

    def test_dataloader(self):
        test_set = FER2013(self.processed_fer2013['x_test'], self.processed_fer2013['y_test'], transform=self.test_transform)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=16)
        return test_loader
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        
        loss = self.criterion(output, labels)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch

        output = self.model(images)
        
        val_loss = self.criterion(output, labels)
        
        preds = output.argmax(dim=1, keepdim=True)
        corrects = preds.eq(labels.view_as(preds)).sum().item()
        accuracy = corrects / len(batch)
        
        result = {'val_loss': val_loss, 'corrects': corrects, 'num_labels': labels.shape[0]}
        self.log('val_loss', val_loss)
        return result

    def validation_epoch_end(self, outputs):
        val_losses = [output['val_loss'].item() for output in outputs]
        corrects = [output['corrects'] for output in outputs]
        num_labels = [output['num_labels'] for output in outputs]
        val_acc = np.sum(corrects) / np.sum(num_labels)
        
        self.log('val_loss_epoch', np.mean(val_losses), on_epoch=True, prog_bar=True)
        self.log('val_accuracy', val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        
        output = self.model(images)
        
        test_loss = self.criterion(output, labels)
        
        preds = output.argmax(dim=1, keepdim=True)
        corrects = preds.eq(labels.view_as(preds)).sum().item()
        
        result = {'test_loss': test_loss, 'corrects': corrects, 'num_labels': preds.shape[0]}
        return result
    
    def inference(self, input):
        emotions = {
            0:"Angry",
            1:"Disgust",
            2:"Fear",
            3:"Happy",
            4:"Sad",
            5:"Surprize",
            6:"Neutral"
        }
        transform = transforms.Normalize(mean=0, std = 255)
        input = torch.from_numpy(input)[None, None, :, :].float()
        input = transform(input)
        prediction = emotions[self.model(input).argmax(dim=1).item()]
        return prediction

    def test_epoch_end(self, outputs) -> None:
        test_losses = [output['test_loss'].item() for output in outputs]
        corrects = [output['corrects'] for output in outputs]
        num_labels = [output['num_labels'] for output in outputs]
        test_acc = np.sum(corrects) / np.sum(num_labels)
        
        self.log('test_loss', np.mean(test_losses))
        self.log('test_acc', test_acc)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, 'monitor': 'val_loss'}

def resnet(weights_path):
    print("Loading pretrained ResNet.")
    with open(weights_path, 'rb') as f:
        obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    resnet50 = models.resnet50()
    resnet50.fc = nn.Linear(in_features=2048, out_features=8631, bias=True)
    resnet50.load_state_dict(weights)

    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size = 3, padding=1, bias=False)
    resnet50.maxpool = nn.Sequential()
    resnet50.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1024, bias=True),
        nn.Linear(in_features=1024, out_features=7, bias=True)
    )
    print("Pretrained ResNet Loaded!")
    return resnet50