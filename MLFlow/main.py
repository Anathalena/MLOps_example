import mlflow.pytorch
import lightning.pytorch as pl
from model import Model
from dataloader import Data

if __name__ == '__main__':
    mlflow.pytorch.autolog()
    model = Model(input_size=28*28, hidden_size=100, num_classes=10, learning_rate=1e-3, batch_size=32)
    data = Data(batch_size=32)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=model, datamodule=data)
    trainer.test(datamodule=data)
    trainer.validate(datamodule=data)