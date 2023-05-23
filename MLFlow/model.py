import torch
from torch import nn
import lightning.pytorch as pl



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
        x = x.view(-1,28*28)
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

