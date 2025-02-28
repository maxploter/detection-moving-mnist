import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

from generate import DATASET_SPLITS


def parse_args():
	parser = argparse.ArgumentParser(
		description="Calculate dataset statistics and create distribution histograms"
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
		default=DATASET_SPLITS,
		help=f"Dataset splits to process (e.g., {','.join(DATASET_SPLITS)})"
	)
	return parser.parse_args()


def create_histogram(data, title, xlabel, ylabel, bins, filename):
	plt.figure(figsize=(10, 6))
	counts, bins, _ = plt.hist(data, bins=bins, density=True, alpha=0.75)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xticks(bins)
	plt.grid(True, alpha=0.3)
	plt.savefig(filename)
	plt.close()


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

		# Initialize statistics accumulators
		sum_pixels = None
		sum_squares = None
		total_pixels = 0

		# Initialize histogram counters
		digits_per_frame = []
		digit_counts = defaultdict(int)
		total_digits = 0

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
			# Process video for statistics
			video = example['video'].get_batch(range(len(example["video"]))).asnumpy().astype(np.float32) / 255.0

			if video.ndim == 3:
				video = np.expand_dims(video, axis=-1)

			video_tensor = torch.from_numpy(video)
			sum_pixels += video_tensor.sum(dim=(0, 1, 2))
			sum_squares += (video_tensor ** 2).sum(dim=(0, 1, 2))
			total_pixels += video_tensor.shape[0] * video_tensor.shape[1] * video_tensor.shape[2]

			# Process targets for histograms
			if 'targets' in example:
				for frame_targets in example['targets']:
					if 'labels' in frame_targets:
						num_digits = len(frame_targets['labels'])
						digits_per_frame.append(num_digits)

						for digit in frame_targets['labels']:
							digit_counts[digit] += 1
							total_digits += 1

		# Calculate and store statistics
		mean = sum_pixels / total_pixels
		std = torch.sqrt((sum_squares / total_pixels) - (mean ** 2))

		dataset_stats[split] = {
			"mean": mean.tolist(),
			"std": std.tolist()
		}

		# Create and save histograms
		if digits_per_frame:
			# Digits per frame distribution
			max_digits = max(digits_per_frame)
			plt_path = os.path.join(args.dataset_dir, f"{split}_digits_per_frame.png")
			create_histogram(
				digits_per_frame,
				f"Normalized Digits per Frame Distribution ({split})",
				"Number of Digits",
				"Normalized Frequency",
				bins=np.arange(0, max_digits + 2) - 0.5,
				filename=plt_path
			)

		if digit_counts:
			# Digit class distribution
			digits = sorted(digit_counts.keys())
			counts = np.array([digit_counts[d] for d in digits])

			plt.figure(figsize=(10, 6))
			plt.bar(range(10), [digit_counts.get(d, 0) / total_digits for d in range(10)])
			plt.title(f"Normalized Digit Class Distribution ({split})")
			plt.xlabel("Digit Class")
			plt.ylabel("Normalized Frequency")
			plt.xticks(range(10))
			plt.grid(True, alpha=0.3)
			plt.savefig(os.path.join(args.dataset_dir, f"{split}_digit_classes.png"))
			plt.close()

	# Save statistics to JSON file
	stats_path = os.path.join(args.dataset_dir, "dataset_stats.json")
	with open(stats_path, "w") as f:
		json.dump(dataset_stats, f, indent=2)
	print(f"Saved dataset statistics to {stats_path}")


if __name__ == "__main__":
	args = parse_args()
	main(args)