"""Sanity checks on the cross-validation splits; runs on the metadata only.

    python tests/test_splits.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import (N_FOLDS, PosesDataModule, PosesDataModuleFusion, assign_folds,  # noqa: E402
                  performances, read_sheet, split_folds)

sheet = read_sheet(ROOT / "data" / "HEROES_dataset.xlsx")
# Row indices stand in for images, so each sample can be traced back to its clip
rows = np.arange(len(sheet))


def check(condition, message):
    print("{}  {}".format("ok  " if condition else "FAIL", message))
    assert condition, message


labels = sheet.label.to_numpy()
check(np.bincount(labels).tolist() == [747, 621, 662, 647], "labels come from the emotion field")

left, right, front = performances(sheet)
check(len(front) == 891, "891 performances have all three views")
check((sheet.camera[left] == "L").all() and (sheet.camera[right] == "R").all()
      and (sheet.camera[front] == "F").all(), "side views are the left and right cameras")
for a, b in ((left, front), (right, front)):
    check((sheet.actor.to_numpy()[a] == sheet.actor.to_numpy()[b]).all()
          and (sheet.clip_id.to_numpy()[a] == sheet.clip_id.to_numpy()[b]).all(),
          "side views belong to the same performance")

for split in ("actor", "clip"):
    folds = assign_folds(sheet, split)
    perf_key = sheet.actor.astype(str) + "_" + sheet.clip_id.astype(str)
    check((sheet.groupby(perf_key.to_numpy()).apply(lambda g: folds[g.index].ptp()) == 0).all(),
          "[{}] the three views of a performance share a fold".format(split))
    if split == "actor":
        check((sheet.groupby("actor").apply(lambda g: folds[g.index].ptp()) == 0).all(),
              "[actor] every actor sits in a single fold")

    for test_fold in range(N_FOLDS):
        train_folds, val_fold = split_folds(test_fold)
        check(len({test_fold, val_fold, *train_folds}) == N_FOLDS,
              "[{}] fold {}: train / val / test partition the folds".format(split, test_fold + 1))
        for name, module in (("independent", PosesDataModule(rows, sheet, folds, test_fold)),
                             ("fusion", PosesDataModuleFusion(rows, sheet, folds, test_fold))):
            stage_rows = {stage: np.concatenate(module.sets[stage][1:]) for stage in module.sets}
            perf = {stage: set(perf_key[r]) for stage, r in stage_rows.items()}
            check(not (perf["train"] & perf["test"]) and not (perf["val"] & perf["test"])
                  and not (perf["train"] & perf["val"]),
                  "[{}] fold {} {}: no performance shared between stages".format(split, test_fold + 1, name))
            if split == "actor":
                actors = {stage: set(sheet.actor[r]) for stage, r in stage_rows.items()}
                check(not (actors["train"] & actors["test"]) and not (actors["val"] & actors["test"]),
                      "[actor] fold {} {}: no actor shared with test".format(test_fold + 1, name))

print("\nall split checks passed")
