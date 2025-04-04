import json
import logging
import math
import os
import random

import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MovingMNIST:
    def __init__(
        self,
        trajectory,
        affine_params,
        train,
        path="data",
        num_digits=(
            1,
            2,
        ),  # random choice in the tuple to set number of moving digits
        num_frames=10,  # number of frames to generate
        concat=True,  # if we concat the final results (frames, 1, 28, 28) or a list of frames
    ):
        self.train = train
        self.mnist = MNIST(path, download=True, train=train)
        self.total_data_num = len(self.mnist)
        self.trajectory = trajectory
        self.affine_params = affine_params
        self.num_digits = num_digits
        self.num_frames = num_frames
        self.canvas_width = 128
        self.canvas_height = 128
        self.padding = get_padding(128, 128, 28, 28)  # MNIST images are 28x28
        self.concat = concat

    def random_digit(self, initial_translation):
        """Get a random MNIST digit randomly placed on the canvas"""
        mnist_idx = random.randrange(0, self.total_data_num)
        img, label = self.mnist[mnist_idx]
        img = TF.to_tensor(img)
        pimg = TF.pad(img, padding=self.padding)

        x = initial_translation[0]
        y = initial_translation[1]
        placed_img = TF.affine(pimg, translate=[x, y], angle=0, scale=1, shear=[0])

        nonzero_mask = img > 0

        # Get indices of non-zero pixels (channel, y, x)
        nonzero_indices = nonzero_mask.nonzero()
        y_coords = nonzero_indices[:, 1]
        x_coords = nonzero_indices[:, 2]

        # Find min and max coordinates
        min_x = x_coords.min().item()
        max_x = x_coords.max().item()
        min_y = y_coords.min().item()
        max_y = y_coords.max().item()

        # Calculate width and height
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Convert to center-relative coordinates (center is at (14,14))
        x_min = min_x - 14
        y_min = min_y - 14

        digit_bbox = (x_min, y_min, width, height)  # (x_min, y_min, w, h) center is at mnist digit center

        return placed_img, (x,y), label, mnist_idx, digit_bbox

    def _one_moving_digit(self, initial_translation):
        digit, initial_position, label, mnist_idx, mnist_digit_bbox = self.random_digit(initial_translation)
        traj = self.trajectory(
            label,
            self.affine_params,
            n=self.num_frames - 1,
            padding=self.padding,
            initial_position=initial_position,
        )
        frames, positions = traj(digit)

        digit_bboxes = [mnist_digit_bbox] * self.num_frames

        return torch.stack(frames), positions, label, digit_bboxes, mnist_idx

    def __getitem__(self, i):
        digits = random.choice(self.num_digits)
        initial_digit_translations_overlap_free = translate_digits_overlap_free(self.canvas_width, self.canvas_height, digits)

        moving_digits, positions, all_labels, digit_bboxes, mnist_indices = zip(
            *(self._one_moving_digit(initial_digit_translations_overlap_free[i]) for i in range(digits))
        )
        moving_digits = torch.stack(moving_digits)
        combined_digits = moving_digits.max(dim=0)[0]

        # Coordinate system
        #            (top)
        #              -
        #              |
        #              |
        # (left) - ----0---- + (right) X
        #              |
        #              |
        #              + Y
        #           (bottom)
        top_boundary = -self.canvas_height // 2
        bottom_boundary = math.ceil(self.canvas_height / 2) - 1

        left_boundary = -self.canvas_width // 2
        right_boundary = math.ceil(self.canvas_width / 2) - 1

        targets = []
        for frame_number in range(self.num_frames):
            target = {}

            labels = []
            center_points = []
            bboxes_coco_format = [] # x_min, y_min, w, h
            bboxes_keypoints_coco_format = [] # x, y, visibility
            bboxes_labels = []

            for digit_center_points, label, digit_bbox in zip(positions, all_labels, digit_bboxes):
                cx, cy = digit_center_points[frame_number]

                x_min_mnist_center, y_min_mnist_center, bw, bh = digit_bbox[frame_number] # in mnist digit center coordinates

                x_min = cx + x_min_mnist_center
                y_min = cy + y_min_mnist_center

                x_max = x_min + bw
                y_max = y_min + bh

                # Check if the bounding box is within the canvas boundaries
                if x_max > left_boundary and x_min < right_boundary and y_max > top_boundary and y_min < bottom_boundary:
                    # Adjust the bounding box to fit within the canvas
                    x_min = max(x_min, left_boundary)
                    y_min = max(y_min, top_boundary)
                    x_max = min(x_max, right_boundary)
                    y_max = min(y_max, bottom_boundary)

                    # Convert coordinate system from center to top-left corner
                    x_min_tl = x_min + self.canvas_width // 2
                    y_min_tl = y_min + self.canvas_height // 2
                    width = x_max - x_min
                    height = y_max - y_min

                    bboxes_labels.append(label)
                    bboxes_coco_format.append(
                        (x_min_tl, y_min_tl, width, height))  # (x_min, y_min, width, height) in top-left coords

                    if left_boundary <= cx <= right_boundary and top_boundary <= cy <= bottom_boundary:
                        labels.append(label)
                        center_points.append((cx, cy))
                        # Convert center point to top-left coordinate system
                        cx_tl = cx + self.canvas_width // 2
                        cy_tl = cy + self.canvas_height // 2
                        bboxes_keypoints_coco_format.append((cx_tl, cy_tl, 2))  # 2 means visible
                    else:
                        bboxes_keypoints_coco_format.append((0, 0, 0))  # 0 means not labeled not visible

            target['labels'] = labels
            target['center_points'] = center_points
            target['bboxes'] = bboxes_coco_format
            target['bboxes_labels'] = bboxes_labels
            target['bboxes_keypoints'] = bboxes_keypoints_coco_format

            targets.append(target)
        return (
            (
                combined_digits
                if self.concat
                else [t.squeeze(dim=0) for t in combined_digits.split(1)]
            ),
            targets,
            mnist_indices
        )

    def save(self, directory, num_videos, num_videos_hard, whole_dataset=False, hf_videofolder_format=False, hf_arrow_format=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
        assert not (hf_videofolder_format and hf_arrow_format), "Only one format can be selected."

        mnist_indices_used = set()
        seq_index = 0

        if hf_videofolder_format:
            import cv2
            from src.detection_moving_mnist.utils.utils import create_video_from_frames

            number_of_videos_digits = len(str(num_videos)) + 1

            metadata_path = os.path.join(directory, 'metadata.jsonl')
            with open(metadata_path, 'w') as metadata_file:
                # Process initial num_videos
                for _ in tqdm(range(num_videos), desc="Processing sequences"):
                    frames, targets, mnist_indices = self[0]
                    video_filename = f"{seq_index:0{number_of_videos_digits}d}.mp4"
                    output_path = os.path.join(directory, video_filename)
                    create_video_from_frames(
                        frames=frames.squeeze(1),  # Remove channel dimension
                        output_filename=output_path,
                        frame_rate=10.0,
                        resolution=(128, 128),
                        colormap=cv2.COLORMAP_BONE
                    )
                    metadata_entry = {
                        "file_name": video_filename,
                        "targets": targets
                    }
                    metadata_file.write(json.dumps(metadata_entry) + '\n')
                    mnist_indices_used.update(mnist_indices)
                    seq_index += 1

                # Cover entire MNIST dataset if required
                if whole_dataset and len(mnist_indices_used) < len(self.mnist):
                    initial_covered = len(mnist_indices_used)
                    with tqdm(
                        total=len(self.mnist),
                        initial=initial_covered,
                        desc="Covering MNIST dataset"
                    ) as pbar:
                        while len(mnist_indices_used) < len(self.mnist) and seq_index < num_videos_hard:
                            frames, targets, mnist_indices = self[0]
                            video_filename = f"{seq_index:0{number_of_videos_digits}d}.mp4"
                            output_path = os.path.join(directory, video_filename)
                            create_video_from_frames(
                                frames=frames.squeeze(1),
                                output_filename=output_path,
                                frame_rate=10.0,
                                resolution=(128, 128),
                                colormap=cv2.COLORMAP_BONE
                            )
                            metadata_entry = {
                                "file_name": video_filename,
                                "targets": targets
                            }
                            metadata_file.write(json.dumps(metadata_entry) + '\n')
                            prev_covered = len(mnist_indices_used)
                            mnist_indices_used.update(mnist_indices)
                            new_covered = len(mnist_indices_used)
                            pbar.update(new_covered - prev_covered)
                            seq_index += 1

            logging.info(f"Number of used digits: {len(mnist_indices_used)}/{len(self.mnist)}")
            logging.info(f"Video dataset saved to {directory}")
        elif hf_arrow_format:
            from datasets import Dataset, Features, Array3D, Sequence, Value
            import numpy as np

            # Define features for the dataset
            features = Features({
                "video": Array3D(shape=(self.num_frames, 128, 128), dtype="uint8"),
                "labels": Sequence(Sequence(Value("uint8"))),
                "center_points": Sequence(Sequence(Sequence(Value("int16")))),
                "bboxes": Sequence(Sequence(Sequence(Value("int16")))),
                "bboxes_keypoints": Sequence(Sequence(Sequence(Value("int16")))),
                "bboxes_labels": Sequence(Sequence(Value("uint8"))),
            })

            def video_generator():
                nonlocal mnist_indices_used, seq_index
                # Generate initial num_videos
                for _ in range(num_videos):
                    frames, targets, mnist_indices = self[0]

                    # Convert directly to uint8 and remove channel dimension
                    frames_np = (frames.detach().numpy() * 255).astype(np.uint8)
                    frames_np = frames_np.squeeze(axis=1)  # Remove channel dimension
                    # Extract labels and center points
                    labels = [t['labels'] for t in targets]
                    center_points = [t['center_points'] for t in targets]
                    bboxes = [t['bboxes'] for t in targets]
                    bboxes_labels = [t['bboxes_labels'] for t in targets]
                    bboxes_keypoints = [t['bboxes_keypoints'] for t in targets]
                    yield {
                        "video": frames_np,
                        "labels": labels,
                        "center_points": center_points,
                        "bboxes": bboxes,
                        "bboxes_labels": bboxes_labels,
                        "bboxes_keypoints": bboxes_keypoints,
                    }
                    mnist_indices_used.update(mnist_indices)
                    seq_index += 1

                # Generate additional videos if whole_dataset is enabled
                if whole_dataset and len(mnist_indices_used) < len(self.mnist):
                    while True:
                        if num_videos_hard is not None and seq_index >= num_videos_hard:
                            break
                        if len(mnist_indices_used) >= len(self.mnist):
                            break
                        frames, targets, mnist_indices = self[0]

                        # Convert directly to uint8 and remove channel dimension
                        frames_np = (frames.detach().numpy() * 255).astype(np.uint8)
                        frames_np = frames_np.squeeze(axis=1)  # Remove channel dimension
                        labels = [t['labels'] for t in targets]
                        center_points = [t['center_points'] for t in targets]
                        yield {
                            "video": frames_np,
                            "labels": labels,
                            "center_points": center_points,
                        }
                        mnist_indices_used.update(mnist_indices)
                        seq_index += 1

            # Create dataset from generator
            dataset = Dataset.from_generator(
                video_generator,
                features=features,
            )

            # Save the dataset with sharding
            num_shards = 12 if self.train else 2  # Adjust based on split
            dataset.save_to_disk(
                directory,
                num_shards=num_shards,
                storage_options={"compression": "snappy"}
            )

            logging.info(f"Number of used digits: {len(mnist_indices_used)}/{len(self.mnist)}")
            logging.info(f"Arrow-format dataset saved to {directory}")


        else:
            all_targets = []

            for _ in tqdm(range(num_videos), desc="Processing sequences"):
                frames, targets, mnist_indices = self[0]  # Get a single sequence
                torch.save(frames, os.path.join(directory, f"video_{seq_index}_frames.pt"))
                all_targets.append(targets)
                mnist_indices_used.update(list(mnist_indices))
                seq_index += 1

            # Second loop: cover the entire MNIST dataset if required
            if whole_dataset and len(mnist_indices_used) < len(self.mnist):
                initial_covered = len(mnist_indices_used)
                with tqdm(
                    total=len(self.mnist),
                    initial=initial_covered,
                    desc="Covering MNIST dataset"
                ) as pbar:
                    while len(mnist_indices_used) < len(self.mnist):
                        frames, targets, mnist_indices = self[0]
                        torch.save(frames, os.path.join(directory, f"video_{seq_index}_frames.pt"))
                        all_targets.append(targets)
                        prev_covered = len(mnist_indices_used)
                        mnist_indices_used.update(list(mnist_indices))
                        new_covered = len(mnist_indices_used)
                        pbar.update(new_covered - prev_covered)
                        seq_index += 1

            # Save global targets JSON
            with open(os.path.join(directory, "targets.json"), "w") as f:
                json.dump(all_targets, f)

        logging.info(f"Number of used digits from dataset {len(mnist_indices_used)}/{len(self.mnist)}")
        logging.info(f"Tensor-format data saved in directory: {directory}")


def get_padding(target_width, target_height, input_width, input_height):
    """
    Calculate the padding needed to center an image with the given dimensions in a target canvas size.

    Args:
        target_width (int): Target width of the canvas.
        target_height (int): Target height of the canvas.
        input_width (int): Width of the input image.
        input_height (int): Height of the input image.

    Returns:
        tuple: A tuple containing the padding (left_pad, top_pad, right_pad, bottom_pad).
    """
    padding_width = max(target_width - input_width, 0)
    padding_height = max(target_height - input_height, 0)
    left_pad = padding_width // 2
    right_pad = padding_width - left_pad
    top_pad = padding_height // 2
    bottom_pad = padding_height - top_pad
    return left_pad, top_pad, right_pad, bottom_pad


def translate_digits_overlap_free(canvas_width, canvas_height, num_objects, digit_size=28):
    placed_positions = []

    for _ in range(num_objects):
        max_attempts = 20  # Retry limit
        min_overlap_area = 0
        min_overlap_point = None
        for _ in range(max_attempts):
            # Randomly generate a position
            x = random.randint(0, canvas_width - digit_size)
            y = random.randint(0, canvas_height - digit_size)

            overlap_area = 0

            for px, py in placed_positions:
                horizontal_overlap = max(0, min(x + digit_size, px + digit_size) - max(x, px))
                vertical_overlap = max(0, min(y + digit_size, py + digit_size) - max(y, py))
                overlap_area += horizontal_overlap * vertical_overlap

            if overlap_area == 0:
                placed_positions.append((x, y))
                break
            elif min_overlap_point is None or min_overlap_area > overlap_area:
                min_overlap_point = (x, y)
        else:
            assert min_overlap_point is not None
            placed_positions.append(min_overlap_point)

    placed_position_translations = []
    for p in placed_positions:
        x, y = p
        cx, cy = x+digit_size//2, y+digit_size//2
        tx, ty = canvas_width//2 - cx, canvas_height//2 - cy
        placed_position_translations.append((tx, ty))
    return placed_position_translations