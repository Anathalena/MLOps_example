import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from model import Model
from dataloader import Data


if __name__ == '__main__':
    remote_server_ui = 'http://127.0.0.1:5000'
    mlflow.set_tracking_uri(remote_server_ui)

    mlflow.set_experiment("Test")
    mlflow.pytorch.autolog()
    mlflow.start_run(run_name="test_run_v3")

    model = Model(num_classes=10, learning_rate=5e-3, batch_size=32)
    data = Data(batch_size=32)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=model, datamodule=data)
    trainer.validate(datamodule=data)
    trainer.test(datamodule=data)
    