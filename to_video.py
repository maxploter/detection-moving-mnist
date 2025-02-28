import logging
import os
import argparse
import json

import cv2
from tqdm import tqdm

from generate import CONFIGS, DATASET_SPLITS
from src.utils.utils import load_dataset, create_video_from_frames

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert torch-tensor-format to huggingface videofolder format."
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

    output_path_folder = f"mmnist-dataset/huggingface-videofolder-format/mmnist-{version}/{args.split}"

    source_path_folder = f"mmnist-dataset/torch-tensor-format/mmnist-{version}/{args.split}"
    video_frames, video_file_names = load_dataset(source_path_folder)
    number_of_videos = len(video_frames)

    with open(os.path.join(source_path_folder, 'targets.json'), 'r') as f:
        targets_data = json.load(f)

    assert len(targets_data) == number_of_videos, "Each video has its own targets."

    os.makedirs(output_path_folder, exist_ok=True)

    # Process videos and create metadata
    metadata_path = os.path.join(output_path_folder, 'metadata.jsonl')
    with open(metadata_path, 'w') as metadata_file:
      for frames, file_name in tqdm(zip(video_frames, video_file_names), desc="Processing videos"):
        # Extract video index from filename
        parts = file_name.split('_')
        video_index = int(parts[1])

        # Generate MP4 video
        mp4_filename = f"{video_index:0{len(str(number_of_videos-1))}d}.mp4"
        print(mp4_filename)
        output_video_path = os.path.join(output_path_folder, mp4_filename)
        create_video_from_frames(
          frames,
          output_filename=output_video_path,
          frame_rate=10.0,
          resolution=(128, 128),
          colormap=cv2.COLORMAP_BONE,
        )

        # Prepare metadata entry
        metadata_entry = {
          "file_name": mp4_filename,
          "targets": targets_data[video_index]
        }
        metadata_file.write(json.dumps(metadata_entry) + '\n')

    logging.info(f"Dataset saved to {output_path_folder} with metadata.jsonl")


if __name__ == "__main__":
    args = parse_args()
    main(args)
