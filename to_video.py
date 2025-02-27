import logging
import os
import argparse
import shutil

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

    source_path_folder = f"mmnist-dataset/torch-tensor-format/mmnist-{version}/{args.split}"
    video_frames, video_file_names = load_dataset(source_path_folder)

    for i, (video_frames, video_file_name) in enumerate(tqdm(zip(video_frames, video_file_names), desc="Processing videos")):
        create_video_from_frames(
            video_frames,
            output_filename=os.path.join(
                output_path_folder, video_file_name.replace('.pt', '.mp4')
            ),
            frame_rate=10.0,
            resolution=(128, 128),
            colormap=cv2.COLORMAP_BONE,
        )

    shutil.copy2(
        os.path.join(source_path_folder, 'targets.json'),
        os.path.join(output_path_folder, 'targets.json'),
    )

    logging.info(f"Video-format data saved in directory: {output_path_folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
