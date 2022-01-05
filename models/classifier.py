import pytorch_lightning as pl
from torch import nn
import torch
from models.cnn import CNN
import torchmetrics


class Classifier(pl.LightningModule):
    def __init__(self, in_channels, layer_1_dim, layer_2_dim, lr, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = CNN(in_channels, layer_1_dim, layer_2_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = lr

    def forward(self, x):
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        # Let's return preds to use it in a custom callback
        return preds

    # def validation_epoch_end(self, validation_step_outputs):
    #     for pred in validation_step_outputs:

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        outputs = self(x)
        preds = torch.argmax(outputs, dim=1)
        loss = self.loss(outputs, y)
        acc = self.accuracy(preds, y)
        return preds, loss, acc

    def get_layers(self):
        return get_layers_seq(self.classifier.children())


def get_layers_seq(module):
    l = []
    for layer in module:
        if isinstance(layer, nn.Sequential):
            l += get_layers_seq(layer)
        else:
            l.append(layer)
    return l
