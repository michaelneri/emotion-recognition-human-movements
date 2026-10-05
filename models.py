"""Emotion classifiers operating on sequences of binary skeleton images."""

import torch
import torchmetrics
from pytorch_lightning import LightningModule


class _HERBase(LightningModule):
    """Metrics, optimiser, and logging shared by the single-view and fusion models."""

    def __init__(self, lr, num_classes, dropout, weight_decay, class_weights, patience):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience

        self.avgpool = torch.nn.AvgPool3d(kernel_size=(3, 2, 2))
        self.maxpool = torch.nn.MaxPool3d(kernel_size=(3, 2, 2))
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.drop = torch.nn.Dropout3d(self.dropout)
        self.elu = torch.nn.ELU()

        weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.loss_function = torch.nn.CrossEntropyLoss(weight=weight)

        for stage in ("train", "val", "test"):
            setattr(self, "acc_{}".format(stage),
                    torchmetrics.Accuracy(task="multiclass", num_classes=num_classes))
            setattr(self, "auroc_{}".format(stage),
                    torchmetrics.AUROC(task="multiclass", num_classes=num_classes))
            setattr(self, "f1_{}".format(stage),
                    torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro"))

    def loss(self, out, label):
        return self.loss_function(out, label)

    def _log_metrics(self, out, labels_batch, loss, stage):
        for metric in ("acc", "auroc", "f1"):
            obj = getattr(self, "{}_{}".format(metric, stage))
            obj(out, labels_batch)
            self.log("{}_{}".format(stage, metric), obj, on_step=False, on_epoch=True, prog_bar=True)
        self.log("{}_loss".format(stage), loss, on_step=False, on_epoch=True)

    def _step(self, batch, stage):
        out = self(*batch[:-1])
        labels_batch = batch[-1]
        loss = self.loss(out, labels_batch)
        self._log_metrics(out, labels_batch, loss, stage)
        return loss, out, labels_batch

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        _, out, labels_batch = self._step(batch, "test")
        self.test_outputs.append((out.detach().cpu(), labels_batch.detach().cpu()))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="max", patience=self.patience, factor=0.5),
                "monitor": "val_acc",
                "frequency": 1,
            },
        }


class HERModel(_HERBase):
    """Single-view classifier: a 3D CNN over the skeleton image sequence."""

    def __init__(self, lr=0.001, num_classes=4, dropout=0.2, layers=(16, 64, 256),
                 weight_decay=0.0, class_weights=None, patience=50):
        super().__init__(lr, num_classes, dropout, weight_decay, class_weights, patience)
        self.save_hyperparameters()
        self.layers = layers

        self.conv1 = torch.nn.Conv3d(1, layers[0], kernel_size=3, bias=False, padding='same')
        self.ln1 = torch.nn.BatchNorm3d(layers[0])
        self.conv2 = torch.nn.Conv3d(layers[0], layers[1], kernel_size=3, bias=False, padding='same')
        self.ln2 = torch.nn.BatchNorm3d(layers[1])
        self.conv3 = torch.nn.Conv3d(layers[1], layers[2], kernel_size=3, bias=False, padding='same')
        self.ln3 = torch.nn.BatchNorm3d(layers[2])
        self.conv4 = torch.nn.Conv3d(layers[2], num_classes, kernel_size=1, bias=False, padding='same')
        self.ln4 = torch.nn.BatchNorm3d(num_classes)

    def forward(self, x):
        b = x.size(0)
        if self.training:
            # Horizontal flip augmentation, applied per clip
            for i in range(b):
                if torch.rand(1) > 0.5:
                    x[i] = x[i].flip([2])

        x = x.unsqueeze(1)
        x = self.drop(self.elu(self.ln1(self.conv1(x))))
        x = self.maxpool(x) + self.avgpool(x)
        x = self.drop(self.elu(self.ln2(self.conv2(x))))
        x = self.maxpool(x) + self.avgpool(x)
        x = self.drop(self.elu(self.ln3(self.conv3(x))))
        x = self.maxpool(x) + self.avgpool(x)
        x = self.ln4(self.conv4(x))
        return self.gap(x).reshape(b, self.num_classes)


class HERModelLateFusion(_HERBase):
    """Three independent branches (left, right, frontal) averaged at the logit level.

    Each branch is also supervised on its own, so the training loss is the sum
    of the cross-entropy of the three branches and of their average.
    """

    def __init__(self, lr=0.001, num_classes=4, dropout=0.2, layers=(16, 64, 256),
                 weight_decay=0.0, class_weights=None, patience=150, augment=True):
        super().__init__(lr, num_classes, dropout, weight_decay, class_weights, patience)
        self.save_hyperparameters()
        self.layers = layers
        self.augment = augment

        for view in ("L", "R", "C"):
            setattr(self, "conv_{}1".format(view),
                    torch.nn.Conv3d(1, layers[0], kernel_size=3, bias=False, padding='same'))
            setattr(self, "ln_{}1".format(view), torch.nn.BatchNorm3d(layers[0]))
            setattr(self, "conv_{}2".format(view),
                    torch.nn.Conv3d(layers[0], layers[1], kernel_size=3, bias=False, padding='same'))
            setattr(self, "ln_{}2".format(view), torch.nn.BatchNorm3d(layers[1]))
            setattr(self, "conv_{}3".format(view),
                    torch.nn.Conv3d(layers[1], layers[2], kernel_size=3, bias=False, padding='same'))
            setattr(self, "ln_{}3".format(view), torch.nn.BatchNorm3d(layers[2]))
            setattr(self, "conv_{}4".format(view),
                    torch.nn.Conv3d(layers[2], num_classes, kernel_size=1, bias=False, padding='same'))
            setattr(self, "ln_{}4".format(view), torch.nn.BatchNorm3d(num_classes))

    def _branch(self, x, view):
        b = x.size(0)
        x = x.unsqueeze(1)
        for depth in (1, 2, 3):
            x = self.drop(self.elu(
                getattr(self, "ln_{}{}".format(view, depth))(
                    getattr(self, "conv_{}{}".format(view, depth))(x))))
            x = self.maxpool(x) + self.avgpool(x)
        x = getattr(self, "ln_{}4".format(view))(getattr(self, "conv_{}4".format(view))(x))
        return self.gap(x).reshape(b, self.num_classes)

    def forward_branches(self, x_L, x_R, x_C):
        if self.training and self.augment:
            # The three views of a performance are flipped together
            for i in range(x_L.size(0)):
                if torch.rand(1) > 0.5:
                    x_L[i] = x_L[i].flip([2])
                    x_R[i] = x_R[i].flip([2])
                    x_C[i] = x_C[i].flip([2])
        return self._branch(x_L, "L"), self._branch(x_R, "R"), self._branch(x_C, "C")

    def forward(self, x_L, x_R, x_C):
        out_L, out_R, out_C = self.forward_branches(x_L, x_R, x_C)
        return (out_L + out_R + out_C) / 3

    def training_step(self, batch, batch_idx):
        x_L, x_R, x_C, labels_batch = batch
        out_L, out_R, out_C = self.forward_branches(x_L, x_R, x_C)
        out = (out_L + out_R + out_C) / 3
        loss = (self.loss(out, labels_batch) + self.loss(out_L, labels_batch)
                + self.loss(out_R, labels_batch) + self.loss(out_C, labels_batch))
        self._log_metrics(out, labels_batch, loss, "train")
        return loss
