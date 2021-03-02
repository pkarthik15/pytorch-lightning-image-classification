import torch
from torch import nn, optim
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from config import learning_rate, total_number_of_classes, pre_trained


class ClassificationModel(pl.LightningModule):

    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.net = models.resnet50(pretrained=pre_trained)
        self.net.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=total_number_of_classes)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):

        # Output from Dataloader
        imgs, labels = batch

        # Prediction
        preds = self.forward(imgs)

        # Calc Loss
        loss = nn.CrossEntropyLoss()(preds, labels)

        # Calc Accuracy
        acc = accuracy(preds, labels)

        logs = {
            'loss': loss,
            'accuracy': acc
        }

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['logs']['loss'] for x in outputs]).mean()
        avg_accu = torch.tensor([x['logs']['accuracy'] for x in outputs]).mean()
        self.log('validation loss', avg_loss, logger=True, prog_bar=True)
        self.log('validation accuracy', avg_accu, logger=True, prog_bar=True)

    def training_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['logs']['loss'] for x in outputs]).mean()
        avg_accu = torch.tensor([x['logs']['accuracy'] for x in outputs]).mean()
        self.log('training loss', avg_loss, logger=True, prog_bar=True)
        self.log('training accuracy', avg_accu, logger=True, prog_bar=True)
