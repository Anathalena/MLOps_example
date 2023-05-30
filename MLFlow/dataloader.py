from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning.pytorch as pl

class Data(pl.LightningDataModule):
    def __init__(self, batch_size, path='./data'):
        super().__init__()
        self.batch_size = batch_size
        self.transform=transforms.Compose([transforms.ToTensor()]) 
        self.path = path

    def prepare_data(self): 
        MNIST('./data', train=False, download=True)
        MNIST('./data', train=True, download=True)

    def setup(self, stage: str):
        if stage=='test':
            self.test_dataset = MNIST('./data', train=False, transform=self.transform)
        dataset = MNIST('./data', train=True, transform=self.transform)
        train_set_size = int(len(dataset)*0.8)
        val_set_size = len(dataset)-train_set_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_set_size, val_set_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=16)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=16)