import logging
import os
import sys

import cv2
from tqdm import tqdm

from src.utils.utils import load_dataset, create_video_from_frames

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 to_video.py [version]")
        sys.exit(1)
    version = sys.argv[1]

    output_path_folder = f"mmnist-dataset/video-format/mmnist-{version}"

    frame_batches, caption_batches = load_dataset(
        f"mmnist-dataset/torch-tensor-format/mmnist-{version}"
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
    main()
