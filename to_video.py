import logging
import os
import argparse
import json

import cv2
from tqdm import tqdm

from generate import CONFIGS, DATASET_SPLITS
from src.utils.utils import create_video_from_frames, load_dataset

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
	parser.add_argument('--in_place', action='store_true',
	                    help='Remove source files during conversion to save space')

	return parser.parse_args()


def main(args):
	version = args.version
	output_path_folder = f"mmnist-dataset/huggingface-videofolder-format/mmnist-{version}/{args.split}"
	source_path_folder = f"mmnist-dataset/torch-tensor-format/mmnist-{version}/{args.split}"

	# Get list of files first to count videos
	frame_files = sorted([f for f in os.listdir(source_path_folder) if f.endswith("_frames.pt")])
	number_of_videos = len(frame_files)
	number_of_digits = len(str(number_of_videos - 1))

	# Load targets once
	with open(os.path.join(source_path_folder, 'targets.json'), 'r') as f:
		targets_data = json.load(f)
	assert len(targets_data) == number_of_videos, "Each video has its own targets."

	os.makedirs(output_path_folder, exist_ok=True)

	metadata_path = os.path.join(output_path_folder, 'metadata.jsonl')
	with open(metadata_path, 'w') as metadata_file:
		for frames, file_name in tqdm(load_dataset(source_path_folder), total=number_of_videos, desc="Processing videos"):
			# Extract video index from filename
			parts = file_name.split('_')
			video_index = int(parts[1])

			mp4_filename = f"{video_index:0{number_of_digits}d}.mp4"
			output_video_path = os.path.join(output_path_folder, mp4_filename)

			create_video_from_frames(
				frames,
				output_filename=output_video_path,
				frame_rate=10.0,
				resolution=(128, 128),
				colormap=cv2.COLORMAP_BONE,
			)

			# Delete source file if flag set
			if args.in_place:
				source_file = os.path.join(source_path_folder, file_name)
				try:
					os.remove(source_file)
					logging.debug(f"Deleted source file: {source_file}")
				except Exception as e:
					logging.error(f"Error deleting {source_file}: {str(e)}")

			# Write metadata using current video index
			metadata_entry = {
				"file_name": mp4_filename,
				"targets": targets_data[video_index]
			}
			metadata_file.write(json.dumps(metadata_entry) + '\n')

	# Clean up remaining source files if flag set
	if args.in_place:
		try:
			targets_path = os.path.join(source_path_folder, 'targets.json')
			if os.path.exists(targets_path):
				os.remove(targets_path)
				logging.info(f"Deleted source targets.json")

			# Try to remove the now-empty directory
			os.rmdir(source_path_folder)
			logging.info(f"Removed empty source directory: {source_path_folder}")
		except Exception as e:
			logging.error(f"Error cleaning up source directory: {str(e)}")

	logging.info(f"Dataset saved to {output_path_folder} with metadata.jsonl")


if __name__ == "__main__":
	args = parse_args()
	main(args)