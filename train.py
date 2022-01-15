import os

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
import pytorch_lightning as pl
import argparse
from data.datamodule import MNISTDataModule
from milp.milp import MILPModel
from models.classifier import Classifier
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import dict_hash

seed_everything(42, workers=True)

BATCH_SIZE = 128
LAYER_1_DIM = 16
LAYER_2_DIM = 128
LR = 0.09964


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('-l', '--learning_rate', type=float, default=LR, metavar='F',
                        help='learning rate for training (default: 0.09964)')
    parser.add_argument('--layer_1_dim', type=int, default=LAYER_1_DIM, metavar='N',
                        help='dimension of first layer (default: 16)')
    parser.add_argument('--layer_2_dim', type=int, default=LAYER_2_DIM, metavar='N',
                        help='dimension of second layer (default: 128)')
    args = parser.parse_args()
    return vars(args)


def milp_test(model, dataloader, check_models_match=False):
    correct = 0
    tot = 0
    correct_nn = 0
    for x_batch, y_batch in dataloader:
        for x, y in zip(x_batch, y_batch):
            milp_model = MILPModel(model, x, y, 10, include_last_layer=True)
            output_milp = milp_model.optimize()
            y_pred_milp = np.argmax(output_milp)
            if check_models_match:
                output_nn = model(x.unsqueeze(0)).detach().numpy()
                ok = np.allclose(output_milp, output_nn)
                y_pred_nn = np.argmax(output_nn)
                correct_nn += 1 if y_pred_nn == y else 0
                if not ok:
                    print("The two models don't match")
                    print(f"pred_nn: {y_pred_nn}; pred_milp:{y_pred_milp}")
                    print(f"output_nn: {output_nn}")
                    print(f"output_milp: {output_milp}")
                else:
                    print("The two models match")
            tot += 1
            if y == y_pred_milp:
                correct += 1
        accuracy_milp = correct / tot
        print("---------- RESULTS ------------")
        if check_models_match:
            acc_nn = correct_nn / tot
            print(f"PyTorch model accuracy over first batch: {accuracy_milp}")
        print(f"MILP model accuracy over first batch: {accuracy_milp}")
        return accuracy_milp


def train(run_id, layer_1_dim, layer_2_dim, lr, batch_size):
    # set up W&B logger
    wandb_logger = WandbLogger(project='Adversarial Learning', entity="davoli", log_model="all", name=run_id,
                               id=run_id)  # log all
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoint/{run_id}", filename="best", monitor='val_accuracy',
                                          mode='max', save_last=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode="max")

    # setup data
    mnist = MNISTDataModule(batch_size=batch_size)

    # setup model - note how we refer to sweep parameters with wandb.config
    model = Classifier(
        in_channels=1,
        layer_1_dim=layer_1_dim,
        layer_2_dim=layer_2_dim,
        lr=lr
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=20,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=True,
        log_every_n_steps=1
    )

    # train
    last_ckpt_path = f"checkpoint/{run_id}/last.ckpt"
    if os.path.exists(last_ckpt_path):
        trainer.fit(model, datamodule=mnist, ckpt_path=last_ckpt_path)
    else:
        trainer.fit(model, datamodule=mnist)

    best_ckpt_path = f"checkpoint/{run_id}/best.ckpt"
    model.load_from_checkpoint(best_ckpt_path)
    accuracy = milp_test(model, mnist.test_dataloader(), check_models_match=True)
    # trainer.test(model, datamodule=mnist, ckpt_path="best")
    return model


if __name__ == '__main__':
    args = get_args()
    run_hash = dict_hash(args)
    model = train(run_hash, args["layer_1_dim"], args["layer_2_dim"], args["learning_rate"], args["batch_size"])
