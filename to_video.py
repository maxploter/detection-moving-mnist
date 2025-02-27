import logging
import os
import argparse

import cv2
from tqdm import tqdm

from generate import CONFIGS, DATASET_SPLITS
from src.utils.utils import load_dataset, create_video_from_frames

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert torch-tensor-format to video-format."
    )
    parser.add_argument(
        "--version",
        type=str,
        help=f"MMNIST version: {', '.join(CONFIGS.keys())}",
    )
    parser.add_argument(
        "--split",
        type=str,
        help=f"Dataset splits: {', '.join(DATASET_SPLITS)}",
    )

    args = parser.parse_args()

    return args


def main(args):
    version = args.version

    output_path_folder = f"mmnist-dataset/video-format/mmnist-{version}/{args.split}"

    frame_batches, caption_batches = load_dataset(
        f"mmnist-dataset/torch-tensor-format/mmnist-{version}/{args.split}"
    )

    for i, batch in enumerate(tqdm(frame_batches, desc="Processing batches")):
        for j, frames in enumerate(batch):
            create_video_from_frames(
                frames,
                output_filename=os.path.join(
                    output_path_folder, f"batch_{i}_video_{j}.mp4"
                ),
                frame_rate=10.0,
                resolution=(128, 72),
                colormap=cv2.COLORMAP_BONE,
            )

            with open(
                os.path.join(output_path_folder, f"batch_{i}_video_{j}_captions.txt"),
                "w",
            ) as file:
                for line in caption_batches[i][j]:
                    file.write(line + "\n")

    logging.info(f"Video-format data saved in directory: {output_path_folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
