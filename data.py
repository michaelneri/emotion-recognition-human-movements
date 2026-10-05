"""Metadata parsing, cross-validation folds, and data modules for the HEROES clips."""

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

CLASSES = ('Interest', 'Happiness', 'Boredom', 'Disgust')
LABEL_TO_INT = {'I': 0, 'H': 1, 'B': 2, 'D': 3}
N_FOLDS = 5


def read_sheet(path):
    """Load HEROES_dataset.xlsx and parse the clip names.

    Names follow <actor>_<camera>_<emotion>_<clip>_<scenario>.mp4, with camera
    in F/L/R, emotion in I/H/B/D, and scenario A (observational) or B
    (interaction with another person).
    """
    sheet = pd.read_excel(path)
    fields = sheet.FileName.str.replace(".mp4", "", regex=False).str.split("_")
    sheet["actor"] = sheet.id_persona
    sheet["clip_id"] = sheet.id_clip_persona
    sheet["camera"] = fields.str[1]
    sheet["emotion"] = fields.str[2]
    sheet["label"] = sheet.emotion.map(LABEL_TO_INT)
    return sheet


def assign_folds(sheet, split="actor"):
    """Return the fold (0..4) of every clip.

    "actor": actor-independent folds, three actors each; no actor is shared
    between training and test, which is what a model for unseen users must face.
    "clip": the per-clip folds of the sheet, with the three views of a
    performance moved to the fold of its frontal view, so that no performance
    is seen from one camera in training and from another in test.
    """
    if split == "actor":
        actors = sorted(sheet.actor.unique())
        fold_of_actor = {actor: rank % N_FOLDS for rank, actor in enumerate(actors)}
        return sheet.actor.map(fold_of_actor).to_numpy()
    if split == "clip":
        frontal = sheet[sheet.camera == "F"].set_index(["actor", "clip_id"]).Fold
        keys = pd.MultiIndex.from_arrays([sheet.actor, sheet.clip_id])
        return frontal.reindex(keys).to_numpy().astype(int)
    raise ValueError("unknown split {}".format(split))


def performances(sheet):
    """Row indices of the left, right, and frontal view of every performance.

    Performances missing a view are left out.
    """
    views = sheet.reset_index().pivot_table(index=["actor", "clip_id"], columns="camera",
                                            values="index", aggfunc="first")
    views = views.dropna(subset=["L", "R", "F"]).astype(int)
    return views.L.to_numpy(), views.R.to_numpy(), views.F.to_numpy()


def split_folds(test_fold):
    """Validation is the fold after the test one; the remaining three train."""
    val_fold = (test_fold + 1) % N_FOLDS
    train_folds = [f for f in range(N_FOLDS) if f not in (test_fold, val_fold)]
    return train_folds, val_fold


class PosesDataset(torch.utils.data.Dataset):
    """Skeleton-image sequences of one or more views, with their label."""

    def __init__(self, labels, *views):
        self.views = views
        self.labels = labels

    def __getitem__(self, index):
        return (*[torch.tensor(view[index], dtype=torch.float32) for view in self.views],
                torch.tensor(int(self.labels[index]), dtype=torch.long))

    def __len__(self):
        return len(self.labels)


class _FoldDataModule(LightningDataModule):

    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sets = {}

    def _loader(self, stage, shuffle):
        return DataLoader(PosesDataset(*self.sets[stage]), batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)


class PosesDataModule(_FoldDataModule):
    """Single-view clips, restricted to `cameras` (any of F, L, R)."""

    def __init__(self, images, sheet, folds, test_fold, cameras=("F", "L", "R"),
                 batch_size=64, num_workers=0):
        super().__init__(batch_size, num_workers)
        train_folds, val_fold = split_folds(test_fold)
        keep = sheet.camera.isin(cameras).to_numpy()
        labels = sheet.label.to_numpy()
        for stage, members in (("train", np.isin(folds, train_folds)),
                               ("val", folds == val_fold),
                               ("test", folds == test_fold)):
            rows = np.where(members & keep)[0]
            self.sets[stage] = (labels[rows], images[rows])


class PosesDataModuleFusion(_FoldDataModule):
    """Each sample is one performance seen from the left, right, and frontal camera."""

    def __init__(self, images, sheet, folds, test_fold, batch_size=64, num_workers=0):
        super().__init__(batch_size, num_workers)
        train_folds, val_fold = split_folds(test_fold)
        left, right, front = performances(sheet)
        labels = sheet.label.to_numpy()[front]
        perf_folds = folds[front]
        for stage, members in (("train", np.isin(perf_folds, train_folds)),
                               ("val", perf_folds == val_fold),
                               ("test", perf_folds == test_fold)):
            self.sets[stage] = (labels[members], images[left[members]],
                                images[right[members]], images[front[members]])
