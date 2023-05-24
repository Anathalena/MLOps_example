import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import Accuracy


class Model(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, batch_size):
        super().__init__()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()
        self.l1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=1, padding=0),
                                nn.MaxPool2d(2,2),
                                nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=0),
                                nn.MaxPool2d(2,2))
        self.l2 = nn.Sequential(nn.Linear(in_features=48*4*4, out_features=256),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=256, out_features=num_classes))
        
    def forward(self, x):
        x = self.l1(x)
        x = x.view(-1,48*4*4)
        output = self.l2(x)
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        train_loss = self.loss(out,y)
        self.train_accuracy(out,y)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        val_loss = self.loss(out,y)
        self.val_accuracy(out,y)
        self.log("val_acc", self.val_accuracy, on_epoch=True, on_step=False)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.test_accuracy(out,y)
        self.log("test_acc", self.test_accuracy, on_epoch=True, on_step=False)

