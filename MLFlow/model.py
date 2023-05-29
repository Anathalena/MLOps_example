import torch
from torch import nn
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score, f1_score
import mlflow

class Model(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, batch_size):
        super().__init__()
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
        mlflow.log_metric("train_loss", train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        val_loss = self.loss(out,y)
        acc = accuracy_score(y,out)
        mlflow.log_metric("val_acc", acc)
        mlflow.log_metric("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        acc = accuracy_score(y,out)
        f1 = f1_score(y,out)
        mlflow.log_metric("test_acc", acc)
        mlflow.log_metric("f1_score", f1)

