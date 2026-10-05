"""Predict the emotion expressed in a video from body movements only.

    python predict.py clip.mp4
    python predict.py frontal.mp4 --left left.mp4 --right right.mp4

The video goes through the training pipeline: face blurring, Keypoint R-CNN,
skeleton normalisation, and rasterisation. The probabilities are averaged over
the five cross-validation models trained by train.py. The first 20 frames are
skipped and the next 120 (4 s at 30 fps) are analysed.
"""

import argparse
from pathlib import Path

import torch

from build_dataset import skeleton_images
from data import CLASSES
from extract_keypoints import MAXIMUM_FRAME_PER_VIDEO, extract, load_detector, read_clip
from models import HERModel, HERModelLateFusion


def video_to_images(path, detector, device, blur):
    frames = read_clip(path, blur=blur)
    if not frames:
        raise ValueError("{} has fewer than 21 frames".format(path))
    keypoints, valid = extract(frames, detector, device)
    if not valid.any():
        raise ValueError("no person detected in {}".format(path))
    if len(frames) < MAXIMUM_FRAME_PER_VIDEO:
        print("warning: {} provides {} of the {} analysed frames".format(path, len(frames), MAXIMUM_FRAME_PER_VIDEO))
    images = skeleton_images(keypoints[None], valid[None])
    return torch.tensor(images, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="frontal (or single) view")
    parser.add_argument("--left", help="left view, for the late-fusion model")
    parser.add_argument("--right", help="right view, for the late-fusion model")
    parser.add_argument("--checkpoints", help="a .ckpt file or a folder of them; defaults to the "
                        "models trained by `train.py --view frontal` (or `--view fusion`)")
    parser.add_argument("--no-blur", action="store_true", help="skip face blurring")
    args = parser.parse_args()

    fusion = args.left is not None or args.right is not None
    if fusion and (args.left is None or args.right is None):
        parser.error("the late-fusion model needs both --left and --right")
    model_class = HERModelLateFusion if fusion else HERModel
    source = Path(args.checkpoints or "checkpoints/{}_actor".format("fusion" if fusion else "frontal"))
    paths = sorted(source.glob("*.ckpt")) if source.is_dir() else [source]
    if not paths or not paths[0].is_file():
        parser.error("no checkpoint found in {}; train the models first with "
                     "`python train.py --view {}`".format(source, "fusion" if fusion else "frontal"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(device)
    blur = not args.no_blur
    inputs = [video_to_images(args.video, detector, device, blur)]
    if fusion:
        inputs = [video_to_images(args.left, detector, device, blur),
                  video_to_images(args.right, detector, device, blur), inputs[0]]

    probabilities = torch.zeros(len(CLASSES), device=device)
    with torch.no_grad():
        for path in paths:
            model = model_class.load_from_checkpoint(path, map_location=device).eval()
            probabilities += model(*[x.clone() for x in inputs]).softmax(dim=1)[0]
    probabilities /= len(paths)

    print("Prediction: {} ({} model{})".format(CLASSES[int(probabilities.argmax())], len(paths),
                                              "s" if len(paths) > 1 else ""))
    for name, p in sorted(zip(CLASSES, probabilities.tolist()), key=lambda item: -item[1]):
        print("  {:<10} {:6.1%}".format(name, p))


if __name__ == "__main__":
    main()
