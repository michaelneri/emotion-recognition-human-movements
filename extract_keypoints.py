"""Extract COCO-17 body keypoints from videos with Keypoint R-CNN.

Only needed to process new recordings; the keypoints of the HEROES clips are
released in data/. To rebuild them from the original videos:

    python extract_keypoints.py --videos HEROES/ --out data/keypoints_blurred.npz
    python extract_keypoints.py --videos HEROES/ --out data/keypoints.npz --no-blur
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from data import read_sheet

# The first frames of each clip are skipped and the next 120 are kept
SKIP_FRAMES = 20
MAXIMUM_FRAME_PER_VIDEO = 120

_face_cascade = None


def load_detector(device):
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    return model.eval().to(device)


def blur_faces(frame):
    """Anonymise the faces found by a Haar cascade with a strong median blur."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for x, y, w, h in _face_cascade.detectMultiScale(gray, scaleFactor=2.0, minNeighbors=4):
        frame[y:y + h, x:x + w] = cv2.medianBlur(frame[y:y + h, x:x + w], 35)
    return frame


def read_clip(path, blur=True):
    """Return the analysed frames of a video, faces optionally blurred."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError("cannot open {}".format(path))
    frames = []
    index = 0
    success, frame = capture.read()
    while success and index < SKIP_FRAMES + MAXIMUM_FRAME_PER_VIDEO:
        if index >= SKIP_FRAMES:
            frames.append(blur_faces(frame) if blur else frame)
        index += 1
        success, frame = capture.read()
    capture.release()
    return frames


@torch.no_grad()
def extract(frames, detector, device, batch_size=8):
    """Keypoints of the most confident person in every frame.

    Returns `keypoints` (120, 17, 3) with (x, y, visibility) and `valid` (120,),
    False where nobody is detected or the clip is shorter than 120 frames.
    """
    keypoints = np.zeros((MAXIMUM_FRAME_PER_VIDEO, 17, 3), dtype=np.float32)
    valid = np.zeros(MAXIMUM_FRAME_PER_VIDEO, dtype=bool)
    for start in range(0, len(frames), batch_size):
        # Frames are fed in OpenCV's BGR order, as for the released keypoints
        batch = [to_tensor(frame).to(device) for frame in frames[start:start + batch_size]]
        for offset, output in enumerate(detector(batch)):
            if len(output["keypoints"]) > 0:
                keypoints[start + offset] = output["keypoints"][0].cpu().numpy()
                valid[start + offset] = True
    return keypoints, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder with the HEROES .mp4 files")
    parser.add_argument("--sheet", default="data/HEROES_dataset.xlsx")
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--no-blur", action="store_true", help="keep the faces unblurred")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(device)
    sheet = read_sheet(args.sheet)

    all_keypoints = np.zeros((len(sheet), MAXIMUM_FRAME_PER_VIDEO, 17, 3), dtype=np.float32)
    all_valid = np.zeros((len(sheet), MAXIMUM_FRAME_PER_VIDEO), dtype=bool)
    for i, name in enumerate(tqdm(sheet.FileName)):
        frames = read_clip(Path(args.videos) / name, blur=not args.no_blur)
        all_keypoints[i], all_valid[i] = extract(frames, detector, device)
    np.savez_compressed(args.out, keypoints=all_keypoints, valid=all_valid)


if __name__ == "__main__":
    main()
