import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch

from src.mmnist.mmnist import MovingMNIST
from src.mmnist.trajectory import (
    BouncingTrajectory,
    RandomTrajectory,
    SimpleLinearTrajectory,
    OutOfBoundsTrajectory,
)

TRAIN_SPLIT = 'train'
TEST_SPLIT = 'test'
MINI_SPLIT = 'mini'
DATASET_SPLITS = [TRAIN_SPLIT, TEST_SPLIT]

CONFIGS = {
    "easy": {
        "angle": (0, 0),  # No rotation
        "translate": ((-5, 5), (-5, 5)),
        "scale": (1, 1),  # No scaling
        "shear": (0, 0),  # No deformation on z-axis
        "num_digits": (1,2,3,4,5,6,7,8,9,10),
    },
    "medium": {
        "angle": (0, 0),
        "translate": ((-2, 2), (-2, 2)),
        "scale": (1, 1),
        "shear": (0, 0),
        "num_digits": (2,),
    },
    "hard": {
        "angle": (0, 0),
        "translate": ((-2, 2), (-2, 2)),
        "scale": (1, 1),
        "shear": (0, 0),
        "num_digits": (2,),
    },
    "random": {
        "angle": (0, 0),
        "translate": ((-2, 2), (-2, 2)),
        "scale": (1, 1),
        "shear": (0, 0),
        "num_digits": (1, 2, 3),
    },
}
TRAJECTORIES = {
    "easy": SimpleLinearTrajectory,
    "medium": BouncingTrajectory,
    "hard": OutOfBoundsTrajectory,
    "random": RandomTrajectory,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Detection MovingMNIST dataset with specified parameters."
    )
    parser.add_argument(
        "--version",
        type=str,
        help=f"MMNIST version: easy",
    )
    parser.add_argument(
        "--split",
        type=str,
        help=f"Dataset splits: {', '.join(DATASET_SPLITS)}",
    )
    parser.add_argument(
        "--num_frames_per_video", type=int, help="Number of frames per video."
    )
    parser.add_argument("--num_videos", type=int, help="Number of videos.")
    parser.add_argument('--whole_dataset', action='store_true', help='We make sure all MNIST digits are used for the dataset.')
    parser.add_argument("--seed", type=int, default=5561, help="Seed.")

    args = parser.parse_args()

    return args


def main(args):
    version = args.version
    num_frames_per_video = args.num_frames_per_video
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if version != 'easy':
        raise ValueError(f'Version {version} is not yet implemented.')

    if version not in CONFIGS:
        raise ValueError(f"Unsupported MMNIST version: {version}")

    affine_params = SimpleNamespace(**CONFIGS[version])
    trajectory = TRAJECTORIES[version]

    if trajectory is None:
        raise NotImplementedError(f"Trajectory not implemented for version: {version}")

    dataset = MovingMNIST(
        trajectory=trajectory,
        train=True if args.split != TEST_SPLIT else False,
        affine_params=affine_params,
        num_digits=CONFIGS[version]["num_digits"],
        num_frames=num_frames_per_video,
    )
    dataset.save(
        directory=f"mmnist-dataset/torch-tensor-format/mmnist-{version}/{args.split}/",
        num_videos=args.num_videos,
        whole_dataset=args.whole_dataset,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
