import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from src.detection_moving_mnist.mmnist.mmnist import MovingMNIST
from src.detection_moving_mnist.mmnist.trajectory import (
    BouncingTrajectory,
    RandomTrajectory,
    SimpleLinearTrajectory,
    OutOfBoundsTrajectory, NonLinearTrajectory,
)

TRAIN_SPLIT = 'train'
TEST_SPLIT = 'test'
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
        "translate": ((-5, 5), (-5, 5)),
        "scale": (1, 1),
        "shear": (0, 0),
        "num_digits": (1,2,3,4,5,6,7,8,9,10),
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
    "medium": NonLinearTrajectory,
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
    parser.add_argument("--num_videos_hard", type=int, help="Number of videos hard limit used when whole_dataset is set.")
    parser.add_argument('--whole_dataset', action='store_true', help='We make sure all MNIST digits are used for the dataset.')
    parser.add_argument("--seed", type=int, default=5561, help="Seed.")
    parser.add_argument('--hf_videofolder_format', action='store_true', help='Save in Hugging Face video folder format.')
    parser.add_argument('--hf_arrow_format', action='store_true', help='Save in Hugging Face arrow format.')
    parser.add_argument(
        "--enable_ranks",
        action='store_true',
        help="Enable ranks for the dataset. This is useful for training models that require rank information."
    )
    parser.add_argument(
        "--enable_delayed_appearance",
        action='store_true',
        help="Enable delayed appearance of digits in the dataset. This means digits will not appear at the beginning of the video."
    )

    args = parser.parse_args()

    return args


def main(args):
    version = args.version
    num_frames_per_video = args.num_frames_per_video
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if version not in ['easy', 'medium']:
        raise ValueError(f'Version {version} is not yet implemented.')

    if version not in CONFIGS:
        raise ValueError(f"Unsupported MMNIST version: {version}")

    affine_params = SimpleNamespace(**CONFIGS[version])
    trajectory = TRAJECTORIES[version]

    if trajectory is None:
        raise NotImplementedError(f"Trajectory not implemented for version: {version}")

    format_folder = "huggingface-videofolder-format" if args.hf_videofolder_format else "torch-tensor-format"
    format_folder = "huggingface-arrow-format" if args.hf_arrow_format else format_folder
    directory = os.path.join(
        "mmnist-dataset",
        format_folder,
        f"mmnist-{version}",
        args.split
    )

    dataset = MovingMNIST(
        trajectory=trajectory,
        train=True if args.split != TEST_SPLIT else False,
        affine_params=affine_params,
        num_digits=CONFIGS[version]["num_digits"],
        num_frames=num_frames_per_video,
        enable_ranks=args.enable_ranks,
        enable_delayed_appearance=args.enable_delayed_appearance,
    )
    dataset.save(
        directory=directory,
        num_videos=args.num_videos,
        num_videos_hard=args.num_videos_hard,
        whole_dataset=args.whole_dataset,
        hf_videofolder_format=args.hf_videofolder_format,
        hf_arrow_format=args.hf_arrow_format,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
