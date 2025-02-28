import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from datasets import load_dataset


def parse_args():
	parser = argparse.ArgumentParser(
		description="Calculate mean and standard deviation of video dataset"
	)
	parser.add_argument(
		"--dataset_dir",
		type=str,
		required=True,
		help="Root directory of dataset in Hugging Face videofolder format"
	)
	parser.add_argument(
		"--splits",
		nargs='+',
		default=["train", "test"],
		help="Dataset splits to process (e.g., train, val, test)"
	)
	return parser.parse_args()


def main(args):
	dataset_stats = {}

	for split in args.splits:
		print(f"Processing {split} split...")

		try:
			dataset = load_dataset(
				"videofolder",
				data_dir=args.dataset_dir,
				split=split,
				trust_remote_code=True
			)
		except Exception as e:
			print(f"Error loading {split}: {str(e)}")
			continue

		# Initialize accumulators
		sum_pixels = None
		sum_squares = None
		total_pixels = 0

		# Get first video to determine channels
		try:
			first_video = next(iter(dataset))['video']
			num_channels = first_video[-1].asnumpy().shape[-1]
		except StopIteration:
			print(f"No videos found in {split} split")
			continue

		# Re-initialize with proper channel count
		sum_pixels = torch.zeros(num_channels, dtype=torch.float64)
		sum_squares = torch.zeros(num_channels, dtype=torch.float64)

		for example in tqdm(dataset, desc=f"Processing {split} videos"):
			video = example['video'].get_batch(range(len(example["video"]))).asnumpy().astype(np.float32) / 255.0

			# Add channel dimension if missing
			if video.ndim == 3:
				video = np.expand_dims(video, axis=-1)

			# Update accumulators
			video_tensor = torch.from_numpy(video)
			sum_pixels += video_tensor.sum(dim=(0, 1, 2))
			sum_squares += (video_tensor ** 2).sum(dim=(0, 1, 2))
			total_pixels += video_tensor.shape[0] * video_tensor.shape[1] * video_tensor.shape[2]

		# Calculate final statistics
		mean = sum_pixels / total_pixels
		std = torch.sqrt((sum_squares / total_pixels) - (mean ** 2))

		dataset_stats[split] = {
			"mean": mean.tolist(),
			"std": std.tolist()
		}

	# Save statistics to JSON file
	stats_path = os.path.join(args.dataset_dir, "dataset_stats.json")
	with open(stats_path, "w") as f:
		json.dump(dataset_stats, f, indent=2)
	print(f"Saved dataset statistics to {stats_path}")


if __name__ == "__main__":
	args = parse_args()
	main(args)