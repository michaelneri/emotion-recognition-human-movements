"""5-fold cross-validation of the emotion classifier.

    python train.py --view frontal
    python train.py --view fusion
    python train.py --view independent --split clip

For test fold k, fold k+1 is used for validation (model selection) and the
remaining three for training; the test fold is evaluated once, with the
checkpoint that scored best on validation.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from data import N_FOLDS, PosesDataModule, PosesDataModuleFusion, assign_folds, read_sheet
from models import HERModel, HERModelLateFusion

VIEW_CAMERAS = {"frontal": ("F",), "left": ("L",), "right": ("R",), "independent": ("F", "L", "R")}


def run_tag(args):
    return "{}_{}{}{}".format(args.view, args.split,
                              "_noblur" if args.no_blur else "",
                              "_noaug" if args.view == "fusion" and args.no_aug else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", default="frontal",
                        choices=["frontal", "left", "right", "independent", "fusion"],
                        help="a single camera, all cameras pooled, or late fusion of the three")
    parser.add_argument("--split", default="actor", choices=["actor", "clip"],
                        help="actor-independent folds, or the per-clip folds of the sheet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, N_FOLDS + 1)),
                        help="test folds to run, 1-indexed")
    parser.add_argument("--runs", type=int, default=1, help="repetitions, for confidence intervals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-blur", action="store_true", help="use the non-anonymised keypoints")
    parser.add_argument("--no-aug", action="store_true", help="disable horizontal flips (fusion)")
    parser.add_argument("--out", default="results", help="per-fold metrics")
    parser.add_argument("--checkpoints", default="checkpoints", help="best model of every fold")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images = np.load(data_dir / "all_images{}.npy".format("" if args.no_blur else "_blurred"), mmap_mode="r")
    sheet = read_sheet(data_dir / "HEROES_dataset.xlsx")
    folds = assign_folds(sheet, args.split)

    tag = run_tag(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run in range(args.runs):
        for fold in args.folds:
            test_fold = fold - 1
            pl.seed_everything(args.seed + 1000 * run + fold, workers=True)

            if args.view == "fusion":
                data_module = PosesDataModuleFusion(images, sheet, folds, test_fold,
                                                    batch_size=args.batch_size, num_workers=args.workers)
                model = HERModelLateFusion(lr=args.lr, dropout=args.dropout,
                                           weight_decay=args.weight_decay, augment=not args.no_aug)
            else:
                data_module = PosesDataModule(images, sheet, folds, test_fold, VIEW_CAMERAS[args.view],
                                              batch_size=args.batch_size, num_workers=args.workers)
                model = HERModel(lr=args.lr, dropout=args.dropout, weight_decay=args.weight_decay)

            started = time.time()
            with tempfile.TemporaryDirectory() as scratch:
                checkpoint = ModelCheckpoint(dirpath=scratch, monitor="val_acc", mode="max",
                                             save_top_k=1, save_weights_only=True)
                trainer = Trainer(accelerator="auto", devices=1, max_epochs=args.epochs,
                                  logger=False, callbacks=[checkpoint], log_every_n_steps=10,
                                  enable_model_summary=not records)
                trainer.fit(model, datamodule=data_module)

                name = "fold{}.ckpt".format(fold) if args.runs == 1 else "fold{}_run{}.ckpt".format(fold, run)
                shutil.copy(checkpoint.best_model_path, ckpt_dir / name)
                best = type(model).load_from_checkpoint(ckpt_dir / name)
                metrics = trainer.test(best, datamodule=data_module, verbose=False)[0]

            outs = torch.cat([o for o, _ in best.test_outputs]).numpy()
            truth = torch.cat([t for _, t in best.test_outputs]).numpy()
            record = {
                "view": args.view, "split": args.split, "run": run, "fold": fold,
                "n_train": len(data_module.sets["train"][0]),
                "n_val": len(data_module.sets["val"][0]),
                "n_test": len(data_module.sets["test"][0]),
                "best_val_acc": round(float(checkpoint.best_model_score), 4),
                **{k: round(float(v), 4) for k, v in metrics.items() if k != "test_loss"},
                "confusion_matrix": confusion_matrix(truth, outs.argmax(1), labels=range(4)).tolist(),
                "minutes": round((time.time() - started) / 60, 1),
            }
            records.append(record)
            print("[{}] fold {} run {}: test acc {:.4f} | macro F1 {:.4f} | AUROC {:.4f}".format(
                tag, fold, run, record["test_acc"], record["test_f1"], record["test_auroc"]))
            (out_dir / "{}.json".format(tag)).write_text(json.dumps(records, indent=2))

    accuracies = [r["test_acc"] for r in records]
    print("\n{}: test accuracy {:.4f} +- {:.4f} over {} folds x runs".format(
        tag, float(np.mean(accuracies)), float(np.std(accuracies)), len(accuracies)))


if __name__ == "__main__":
    main()
