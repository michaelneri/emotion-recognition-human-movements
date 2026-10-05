"""Rasterise the released keypoints into the skeleton images the classifier consumes.

    python build_dataset.py              # face-blurred keypoints, as in the article
    python build_dataset.py --no-blur    # keypoints extracted from the original videos

Writes data/all_images[_blurred].npy. Takes under a minute on CPU.
"""

import argparse

import cv2
import numpy as np
from tqdm import tqdm

# COCO-17 joints, in the order produced by Keypoint R-CNN
KEYPOINTS_STR = ['nose', 'left_eye', 'right_eye',
                 'left_ear', 'right_ear', 'left_shoulder',
                 'right_shoulder', 'left_elbow', 'right_elbow',
                 'left_wrist', 'right_wrist', 'left_hip',
                 'right_hip', 'left_knee', 'right_knee',
                 'left_ankle', 'right_ankle']

LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


def to_pixels(xy, dim):
    """Map (17, 2) image coordinates to integer pixels of a dim x dim frame.

    The longer side of the bounding box spans the frame and the shorter one is
    centred, so the skeleton is independent of the distance from the camera
    and keeps its proportions.
    """
    low = xy.min(axis=0)
    extent = xy.max(axis=0) - low
    scale = (dim - 1) / max(float(extent.max()), 1e-6)
    offset = (dim - 1 - extent * scale) / 2
    return np.clip(np.floor((xy - low) * scale + offset), 0, dim - 1).astype(int)


def draw_skeleton(pixels, dim):
    """Binary image of the limbs plus virtual head, trunk, and pelvis joints."""
    frame = np.zeros((dim, dim), dtype=np.uint8)
    point = lambda p: (int(p[0]), int(p[1]))  # noqa: E731  (x, y) as OpenCV expects
    head = np.floor(pixels[:5].mean(axis=0))
    trunk = np.floor(pixels[[5, 6]].mean(axis=0))
    pelvis = np.floor(pixels[[11, 12]].mean(axis=0))
    segments = [(head, trunk), (pelvis, trunk), (pixels[5], trunk), (pixels[6], trunk),
                (pixels[11], pelvis), (pixels[12], pelvis)]
    segments += [(pixels[a], pixels[b]) for a, b in LIMBS]
    for start, end in segments:
        cv2.line(frame, point(start), point(end), 1, 1)
    # Body joints, indexed as (row, column) = (y, x) like the lines
    frame[pixels[5:, 1], pixels[5:, 0]] = 1
    return frame


def skeleton_images(keypoints, valid, dim=32):
    """Turn (N, T, 17, 3) keypoints into (N, T, dim, dim) binary skeleton images.

    Frames where `valid` is False stay black. Pixels are 0 or 1 (uint8).
    """
    n_videos, n_frames = valid.shape
    images = np.zeros((n_videos, n_frames, dim, dim), dtype=np.uint8)
    for i in tqdm(range(n_videos), disable=n_videos < 10):
        for f in np.where(valid[i])[0]:
            images[i, f] = draw_skeleton(to_pixels(keypoints[i, f, :, :2], dim), dim)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-blur", action="store_true",
                        help="use the keypoints extracted from the original, non-anonymised videos")
    parser.add_argument("--dim", type=int, default=32, help="side of the square skeleton image")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    suffix = "" if args.no_blur else "_blurred"
    archive = np.load("{}/keypoints{}.npz".format(args.data_dir, suffix))
    images = skeleton_images(archive["keypoints"], archive["valid"], dim=args.dim)
    np.save("{}/all_images{}.npy".format(args.data_dir, suffix), images)


if __name__ == "__main__":
    main()
