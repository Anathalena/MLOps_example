from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning.pytorch as pl

import mlflow.pytorch

mlflow.pytorch.autolog()

class Model(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate, batch_size):
        super().__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()
        self.l1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, num_classes))
        
    def forward(self, x):
        x = x.view(self.batch_size, -1)
        output = self.l1(x)
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        train_loss = self.loss(out,y)
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        val_loss = self.loss(out,y)
        self.log("val_loss", val_loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        test_loss = self.loss(out,y)
        self.log("test_loss", test_loss, on_epoch=True)



class Data(pl.LightningDataModule):
    def __init__(self, batch_size, path='./data'):
        super().__init__()
        self.batch_size = batch_size
        self.transform=transforms.Compose([transforms.ToTensor()]) 
        self.path = path

    def prepare_data(self): 
        self.train_dataset = MNIST('./data', train=True, download=True, transform=self.transform)
        self.test_dataset = MNIST('./data', train=False, download=True, transform=self.transform)
        self.train_set_size = int(len(self.train_dataset))
        self.indices = np.arange(self.train_set_size)
        np.random.shuffle(self.indices)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False, sampler = SubsetRandomSampler(self.indices[0:int(self.train_set_size*0.8)]))
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = False, sampler = SubsetRandomSampler(self.indices[int(self.train_set_size*0.8):]))

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size = self.batch_size)


model = Model(input_size=28*28, hidden_size=100, num_classes=10, learning_rate=1e-3, batch_size=16)
data = Data(batch_size=16)

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, data)