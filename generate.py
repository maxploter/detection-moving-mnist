import argparse
import sys
from types import SimpleNamespace

from src.mmnist.mmnist import MovingMNIST
from src.mmnist.trajectory import (
    BouncingTrajectory,
    RandomTrajectory,
    SimpleLinearTrajectory,
    OutOfBoundsTrajectory,
)

TRAIN_SPLIT = 'train'
TEST_SPLIT = 'test'
DATASET_SPLITS = [TRAIN_SPLIT, TEST_SPLIT]

CONFIGS = {
    "easy": {
        "angle": (0, 0),  # No rotation
        "translate": ((-1, 1), (-1, 1)),
        "scale": (1, 1),  # No scaling
        "shear": (0, 0),  # No deformation on z-axis
        "num_digits": (1,),
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
        "version",
        choices=CONFIGS.keys(),
        help=f"MMNIST version: {', '.join(CONFIGS.keys())}",
    )
    parser.add_argument(
        "split",
        choices=DATASET_SPLITS,
        help=f"Dataset splits: {', '.join(DATASET_SPLITS)}",
    )
    parser.add_argument(
        "num_frames_per_video", type=int, help="Number of frames per video."
    )
    parser.add_argument("num_batches", type=int, help="Number of batches to generate.")
    parser.add_argument("batch_size", type=int, help="Batch size.")

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    version = args.version
    num_frames_per_video = args.num_frames_per_video
    num_batches = args.num_batches
    batch_size = args.batch_size

    if version not in CONFIGS:
        raise ValueError(f"Unsupported MMNIST version: {version}")

    affine_params = SimpleNamespace(**CONFIGS[version])
    trajectory = TRAJECTORIES[version]

    if trajectory is None:
        raise NotImplementedError(f"Trajectory not implemented for version: {version}")

    dataset = MovingMNIST(
        trajectory=trajectory,
        affine_params=affine_params,
        num_digits=CONFIGS[version]["num_digits"],
        num_frames=num_frames_per_video,
    )
    dataset.save(
        directory=f"mmnist-dataset/torch-tensor-format/mmnist-{version}",
        n_batches=num_batches,
        bs=batch_size,
    )


if __name__ == "__main__":
    main()
