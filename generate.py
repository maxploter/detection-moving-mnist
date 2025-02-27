import sys
from types import SimpleNamespace

from src.mmnist.mmnist import MovingMNIST
from src.mmnist.trajectory import (
    BouncingTrajectory,
    RandomTrajectory,
    SimpleLinearTrajectory,
    OutOfBoundsTrajectory,
)

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


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python3 generate.py [version] [num_frames_per_video] [num_batches] [batch_size]"
        )
        sys.exit(1)
    version = sys.argv[1]
    num_frames_per_video = int(sys.argv[2])
    num_batches = int(sys.argv[3])
    batch_size = int(sys.argv[4])

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
