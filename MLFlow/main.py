import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from model import Model
from dataloader import Data


if __name__ == '__main__':
    remote_server_ui = 'http://127.0.0.1:5000'
    mlflow.set_tracking_uri(remote_server_ui)
    mlflow.set_experiment("Test with auto log")
    mlflow.pytorch.autolog(log_models=True)

    mlflow.start_run(run_name="MNIST-classification-testingv2.0")

    model = Model(num_classes=10, learning_rate=2e-3, batch_size=32)
    data = Data(batch_size=32)

    trainer = pl.Trainer(max_epochs=50, callbacks=[EarlyStopping(monitor="val_loss", mode='min', patience=5)])
    trainer.fit(model, datamodule=data)
    
    data.setup(stage='test')
    trainer.test(ckpt_path='best', datamodule=data)
    