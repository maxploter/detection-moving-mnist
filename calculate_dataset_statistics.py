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
		"--dataset",
		type=str,
		required=True,
		help="Path to dataset on filesystem or name of the dataset in Hugging Face hub"
	)
	parser.add_argument(
		"--splits",
		nargs='+',
		default=DATASET_SPLITS,
		help=f"Dataset splits to process (e.g., {','.join(DATASET_SPLITS)})"
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=".",
		help="Directory to save output files"
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

	# Ensure output directory exists
	os.makedirs(args.output_dir, exist_ok=True)

	for split in args.splits:
		print(f"Processing {split} split...")

		try:
			dataset = load_dataset(
				args.dataset,
				split=split,
			)
		except Exception as e:
			print(f"Error loading {split}: {str(e)}")
			continue

		# Initialize statistics accumulators
		total_pixels = 0

		# Initialize histogram counters
		digits_per_frame = []
		digit_counts = defaultdict(int)
		total_digits = 0

		# Get first video to determine channels
		first_example = next(iter(dataset))
		video = np.array(first_example['video'])
		num_channels = 1 if video.ndim == 3 else video.shape[-1]

		# Re-initialize with proper channel count
		sum_pixels = torch.zeros(num_channels, dtype=torch.float64)
		sum_squares = torch.zeros(num_channels, dtype=torch.float64)

		for example in tqdm(dataset, desc=f"Processing {split} videos"):
			# Process video for statistics
			video = np.array(example['video']).astype(np.float32) / 255.0

			if video.ndim == 3:
				video = np.expand_dims(video, axis=-1)

			video_tensor = torch.from_numpy(video)
			sum_pixels += video_tensor.sum(dim=(0, 1, 2))
			sum_squares += (video_tensor ** 2).sum(dim=(0, 1, 2))
			total_pixels += video_tensor.shape[0] * video_tensor.shape[1] * video_tensor.shape[2]

			# Process targets for histograms
			for i in range(video_tensor.shape[0]):
				frame_labels = example['bboxes_labels'][i]
				num_digits = len(frame_labels)
				digits_per_frame.append(num_digits)

				for digit in frame_labels:
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
			# Digits per frame distribution (allow 0 objects)
			max_digits = max(digits_per_frame)
			plt_path = os.path.join(args.output_dir, f"{split}_digits_per_frame.png")
			create_histogram(
				digits_per_frame,
				f"Normalized Digits per Frame Distribution ({split})",
				"Number of Objects",
				"Normalized Frequency",
				bins=np.arange(0, max_digits + 2),  # bins from 0 to max_digits (inclusive)
				filename=plt_path
			)

			# Plot ratio of empty frames vs frames with objects
			num_empty = sum(1 for n in digits_per_frame if n == 0)
			num_nonempty = len(digits_per_frame) - num_empty
			plt.figure(figsize=(6, 6))
			plt.pie(
				[num_empty, num_nonempty],
				labels=["Empty frames", "Frames with objects"],
				autopct='%1.1f%%',
				colors=["#cccccc", "#66b3ff"],
				startangle=90
			)
			plt.title(f"Empty vs Non-empty Frames Ratio ({split})")
			plt.savefig(os.path.join(args.output_dir, f"{split}_empty_vs_nonempty_frames.png"))
			plt.close()

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
			plt.savefig(os.path.join(args.output_dir, f"{split}_digit_classes.png"))
			plt.close()

	# Save statistics to JSON file
	stats_path = os.path.join(args.output_dir, "dataset_stats.json")
	with open(stats_path, "w") as f:
		json.dump(dataset_stats, f, indent=2)
	print(f"Saved dataset statistics to {stats_path}")


if __name__ == "__main__":
	args = parse_args()
	main(args)