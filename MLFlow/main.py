import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from model import Model
from dataloader import Data

import os
os.environ['MLFLOW_TRACKING_USERNAME'] = '<user_name>'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '<pass>'

if __name__ == '__main__':
    remote_server_ui = 'http://<ip>:5000'
    mlflow.set_tracking_uri(remote_server_ui)

    mlflow.set_experiment("Test")
    mlflow.pytorch.autolog()
    mlflow.start_run(run_name="test_run")

    model = Model(num_classes=10, learning_rate=1e-3, batch_size=32)
    data = Data(batch_size=32)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=model, datamodule=data)
    trainer.validate(datamodule=data)
    trainer.test(datamodule=data)
    