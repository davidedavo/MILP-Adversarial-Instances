import json
import os
import time

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
import pytorch_lightning as pl
import argparse

from tqdm import tqdm

from data.datamodule import MNISTDataModule
from milp.milp_model import MILPModel
from models.classifier import Classifier
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

from models.cnn import DNN, CNN
from utils import get_adversarial_label, AdversarialLogger

seed_everything(42, workers=True)

BATCH_SIZE = 32
LR = 0.09964

config = {
    "conv_layers": [],
    "linear_layers": [],
    "kernel_sizes": 19,
    "max_pool2d": 0,
    "lr": LR
}


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--classifier', choices=['CNN_1x4', 'DNN_8x8'], default='CNN_8',
                        help='Classifier name.')
    args = parser.parse_args()
    return vars(args)


def milp_test(model, trainer, dataloader, bounds, global_step, n_adv=3, policy="random", log_images=False):
    diffs = []
    logger = AdversarialLogger(trainer, policy)

    index = 0
    for batch_id, (x_batch, y_batch) in enumerate(dataloader):
        for image_id, (x, y_true) in tqdm(enumerate(zip(x_batch, y_batch))):
            output_nn = model(x.unsqueeze(0))
            y_pred = output_nn.argmax(dim=1)
            x_numpy = x.detach().numpy()
            if y_true != y_pred:
                continue

            adv_images = []
            adv_labels = []
            diffs_example = []

            start_setup = time.time()
            milp_model = MILPModel(model, x_numpy, y_true.item(), 10, bounds)
            milp_model.setup()
            end_setup = time.time()
            setup_time = end_setup - start_setup

            mean_opt_time = 0
            for i in range(n_adv):
                y_target = get_adversarial_label(y_true, output_nn.squeeze(), adv_labels, policy)

                start_opt = time.time()
                milp_model.buildAdversarialProblem(y_target)
                res = milp_model.optimize()
                end_opt = time.time()
                mean_opt_time += (end_opt - start_opt) / n_adv

                if not res:
                    continue

                output_milp, adv, diff = res
                y_pred_milp = np.argmax(output_milp)
                if y_pred_milp != y_target:
                    print(
                        f"Error! y_pred_milp:{y_pred_milp}, y_target:{y_target} for image {image_id} in batch {batch_id}")

                adv_images.append(torch.from_numpy(adv))
                adv_labels.append(y_target)
                diffs_example.append(diff)

            logger.add_times(index, mean_opt_time, setup_time)

            if log_images:
                logger.log_adversarial(index, x, adv_images, adv_labels, y_true, diffs_example)
            diffs.append(min(diffs_example))
            index += 1

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        wandb.run.summary["mean_diff"] = mean_diff
        wandb.run.summary["std_diff"] = std_diff
        logger.log_times()

        return None


def train(run_id, classifier, params, batch_size, force_train=False):
    # set up W&B logger
    if not run_id:
        run_id = wandb.util.generate_id()

    wandb_logger = WandbLogger(project='Adversarial Learning', entity="davoli", log_model="all", name=run_id,
                               id=run_id)  # log all

    wandb_logger.experiment.config.update(params)
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoint/{run_id}", filename="best", monitor='val_accuracy',
                                          mode='max', save_last=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode="max")

    # setup data
    mnist = MNISTDataModule(batch_size=batch_size)

    # setup model - note how we refer to sweep parameters with wandb.config
    model = Classifier(
        classifier=classifier,
        in_channels=1,
        params=params,
        num_classes=10
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
    milp_test(model, trainer, mnist.test_dataloader(), mnist.get_data_bounds(), trainer.global_step)
    # trainer.test(model, datamodule=mnist, ckpt_path="best")
    return model


if __name__ == '__main__':
    args = get_args()
    with open('model_config.json') as json_file:
        model_configs = json.load(json_file)

    config.update(model_configs[args["classifier"]])

    classifier = CNN
    if config["type"] == "DNN":
        classifier = DNN

    model = train(args["classifier"], classifier, config, BATCH_SIZE)
