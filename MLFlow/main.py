import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from model import Model
from dataloader import Data


if __name__ == '__main__':
    remote_server_ui = 'http://127.0.0.1:5000'
    run_name="MNIST classification - testing"
    mlflow.set_tracking_uri(remote_server_ui)
    mlflow.set_experiment("Test with no auto log")

    for lr in [0.001, 0.005, 0.01, 0.1]:
        with mlflow.start_run(run_name=run_name) as run:
        
            model = Model(num_classes=10, learning_rate=lr, batch_size=32)
            data = Data(batch_size=32)

            mlflow.pytorch.log_model(model, "cnn-model")
            mlflow.log_param("learning_rate", lr)

            trainer = pl.Trainer(max_epochs=5)
            trainer.fit(model=model, datamodule=data)
            trainer.validate(datamodule=data)
            trainer.test(datamodule=data)

            print("Run with id {} and experiment id {} is done.".format(run.info.run_uuid, run.info.experiment_id))
    