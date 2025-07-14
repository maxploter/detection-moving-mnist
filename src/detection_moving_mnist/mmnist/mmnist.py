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
        initial_digits_overlap_free=False,  # if we want to place digits overlap free
        enable_ranks=True,
        enable_delayed_appearance=True,
        max_retries=10,  # maximum number of retries for empty target sequences
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
        self.initial_digits_overlap_free = initial_digits_overlap_free
        self.enable_ranks = enable_ranks
        self.enable_delayed_appearance = enable_delayed_appearance
        self.max_retries = max_retries

    def random_digit(self, initial_translation):
        """Get a random MNIST digit randomly placed on the canvas"""
        mnist_idx = random.randrange(0, self.total_data_num)

        x = initial_translation[0]
        y = initial_translation[1]

        return (x,y), mnist_idx

    def _one_moving_digit(self, initial_translation):
        # Get the original digit and its properties
        initial_position, mnist_idx = self.random_digit(initial_translation)

        mnist_img, label = self.mnist[mnist_idx]
        mnist_img = TF.to_tensor(mnist_img)

        if self.enable_delayed_appearance:
            appearance_frame = random.randint(0, self.num_frames // 2)
        else:
            appearance_frame = 0

        trajectory_params = get_trajectory_params(self.trajectory, mnist_label=label)

        traj = self.trajectory(
            label,
            self.affine_params,
            n=self.num_frames,
            padding=self.padding,
            initial_position=initial_position,
            first_appearance_frame=appearance_frame,
            mnist_img=mnist_img,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            **trajectory_params
        )

        trajectory_data = traj()

        return trajectory_data, label, mnist_idx

    def _has_valid_targets(self, targets):
        """Check if at least one frame in the sequence has valid targets."""
        for target in targets:
            if target['labels'] and len(target['labels']) > 0:
                return True
        return False

    def _get_internal(self, i):
        """Generate a sequence of moving digits with their targets."""
        digits = random.choice(self.num_digits)
        initial_digit_translations = translate_digits_overlap_free(self.canvas_width, self.canvas_height,
                                                                   digits) if self.initial_digits_overlap_free else translate_digits_randomly(
            self.canvas_width, self.canvas_height, digits, canvas_multiplier=1.5)

        trajectory_data_per_digit, all_labels, mnist_indices = zip(
            *(self._one_moving_digit(initial_digit_translations[i]) for i in range(digits))
        )

        # Initialize frame tensor
        combined_digits = torch.zeros((self.num_frames, 1, self.canvas_width, self.canvas_height))
        accumulated_masks = torch.zeros((self.num_frames, 1, self.canvas_width, self.canvas_height), dtype=torch.bool)

        # Apply ranking system when combining digits
        targets = []

        for frame_idx in range(self.num_frames):
            target = {}

            labels = []
            center_points = []
            bboxes_coco_format = [] # x_min, y_min, w, h
            bboxes_keypoints_coco_format = [] # x, y, visibility
            bboxes_labels = []
            track_ids = []

            for idx in range(digits - 1, -1, -1):
                mnist_idx = mnist_indices[idx]
                original_digit, label = self.mnist[mnist_idx]
                original_digit = TF.to_tensor(original_digit)
                trajectory_data = trajectory_data_per_digit[idx]

                if frame_idx not in trajectory_data:
                    continue

                x, y = trajectory_data[frame_idx]['center_point']

                # Create empty canvas for this digit
                digit_canvas = trajectory_data[frame_idx]['frame']

                # Create mask for this digit
                digit_mask = torch.zeros_like(digit_canvas, dtype=torch.bool)
                bbox = trajectory_data[frame_idx]['bbox']
                if bbox is None:
                    continue

                (x_min, y_min, width, height) = bbox

                if self.enable_ranks:

                    digit_mask[:, y_min:y_min+height, x_min:x_min+width] = True

                    # Only place digit where accumulated mask is False (no digit yet)
                    placement_mask = digit_mask & ~accumulated_masks[frame_idx]

                    # Place the digit and update the accumulated mask
                    combined_digits[frame_idx][placement_mask] = digit_canvas[placement_mask]
                    accumulated_masks[frame_idx] |= digit_mask

                    # Find the visible pixels for this digit after masking
                    digit_canvas_after_overlap = torch.zeros_like(digit_canvas)
                    digit_canvas_after_overlap[placement_mask] = digit_canvas[placement_mask]
                    visible_pixels = digit_canvas_after_overlap.nonzero()

                    # Check if center point is visible or overlapped
                    cx = x + self.canvas_width // 2
                    cy = y + self.canvas_height // 2

                    # Only add valid center points
                    if 0 <= cx < self.canvas_width and 0 <= cy < self.canvas_height:
                        # Check if center point is overlapped by previously placed digits
                        center_point_visible = placement_mask[0, int(cy), int(cx)].item()
                        # We'll still track the point but mark visibility based on overlap
                        visibility = 2 if center_point_visible else 1
                    else:
                        visibility = 0  # Point is outside canvas


                    if visible_pixels.size(0) > 0:  # Check if there are any visible pixels
                        # Get y, x coordinates of visible pixels
                        y_coords = visible_pixels[:, 1]
                        x_coords = visible_pixels[:, 2]

                        # Find bounding box of visible pixels
                        min_x = x_coords.min().item()
                        max_x = x_coords.max().item()
                        min_y = y_coords.min().item()
                        max_y = y_coords.max().item()

                        # Calculate width and height
                        width = max_x - min_x + 1
                        height = max_y - min_y + 1

                        # Store updated bounding box
                        trajectory_data[frame_idx]['bbox'] = (min_x, min_y, width, height)
                    else:
                        # No visible pixels, set empty bbox
                        trajectory_data[frame_idx]['bbox'] = None
                else:
                    # Original max-based combination
                    # combined_digits = moving_digits.max(dim=0)[0]
                    raise NotImplementedError("Ranking system is not enabled, but it is required for this dataset.")

                if trajectory_data[frame_idx]['bbox'] is None:
                    continue

                labels.append(label)
                bboxes_labels.append(label)
                bboxes_coco_format.append(trajectory_data[frame_idx]['bbox'])
                center_points.append((x, y))
                track_ids.append(idx)

                cx = x + self.canvas_width // 2
                cy = y + self.canvas_height // 2

                if cx < 0 or cy < 0 or cx >= self.canvas_width or cy >= self.canvas_height:
                    # If the center point is outside the canvas, we do not add it
                    bboxes_keypoints_coco_format.append((0, 0, 0))
                else:
                    # Convert center point to top-left coordinate system
                    bboxes_keypoints_coco_format.append((cx, cy, visibility))

            target['labels'] = labels
            target['center_points'] = center_points
            target['bboxes'] = bboxes_coco_format
            target['bboxes_labels'] = bboxes_labels
            target['bboxes_keypoints'] = bboxes_keypoints_coco_format
            target['track_ids'] = track_ids

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

    def __getitem__(self, i):
        """
        Get a sequence with at least one valid target. Will retry up to max_retries times.
        """
        for attempt in range(self.max_retries):
            frames, targets, mnist_indices = self._get_internal(i)
            if self._has_valid_targets(targets):
                return frames, targets, mnist_indices

            if attempt + 1 < self.max_retries:
                logging.debug(f"Generated sequence has no valid targets. Retrying ({attempt+1}/{self.max_retries})...")

        logging.warning(f"Failed to generate sequence with valid targets after {self.max_retries} attempts. "
                        f"Returning sequence without valid targets.")
        return frames, targets, mnist_indices

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
                "center_points": Sequence(Sequence(Sequence(Value("float32")))),
                "bboxes": Sequence(Sequence(Sequence(Value("float32")))),
                "bboxes_keypoints": Sequence(Sequence(Sequence(Value("float32")))),
                "bboxes_labels": Sequence(Sequence(Value("uint8"))),
                "track_ids": Sequence(Sequence(Value("uint8"))),
            })

            def video_generator():
                nonlocal mnist_indices_used, seq_index
                # Generate initial num_videos
                for _ in range(num_videos):
                    try:
                        frames, targets, mnist_indices = self[0]
                    except Exception as e:
                        logging.error(f"Error generating video sequence: {e}")
                        raise e

                    # Convert directly to uint8 and remove channel dimension
                    frames_np = (frames.detach().numpy() * 255).astype(np.uint8)
                    frames_np = frames_np.squeeze(axis=1)  # Remove channel dimension
                    # Extract labels and center points
                    labels = [t['labels'] for t in targets]
                    center_points = [t['center_points'] for t in targets]
                    bboxes = [t['bboxes'] for t in targets]
                    bboxes_labels = [t['bboxes_labels'] for t in targets]
                    bboxes_keypoints = [t['bboxes_keypoints'] for t in targets]
                    track_ids = [t['track_ids'] for t in targets]
                    yield {
                        "video": frames_np,
                        "labels": labels,
                        "center_points": center_points,
                        "bboxes": bboxes,
                        "bboxes_labels": bboxes_labels,
                        "bboxes_keypoints": bboxes_keypoints,
                        "track_ids": track_ids,
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

def translate_digits_randomly(canvas_width, canvas_height, num_objects, digit_size=28, canvas_multiplier=1.0):
    placed_positions = []
    for _ in range(num_objects):
        # Randomly generate a position
        x = random.randint(0, canvas_width*canvas_multiplier - digit_size//2)
        y = random.randint(0, canvas_height*canvas_multiplier - digit_size//2)
        placed_positions.append((x, y))

    placed_position_translations = []
    for p in placed_positions:
        x, y = p
        cx, cy = x+digit_size//2, y+digit_size//2
        tx, ty = canvas_width//2 - cx, canvas_height//2 - cy
        placed_position_translations.append((tx, ty))
    return placed_position_translations

def get_trajectory_params(trajectory, mnist_label):
    if trajectory.__name__ == 'NonLinearTrajectory':
        path_type = get_path_type(mnist_label)

        params = {
            "path_type": path_type,
            "amplitude": random.uniform(1, 10),
            "frequency": random.uniform(0.05, 0.25),
        }

        # Add trajectory-specific parameters
        if path_type == "sine":
            params.update({
                "phase": random.uniform(0, 2 * math.pi),  # Phase shift for the sine wave
                "direction": random.choice(["vertical"]),  # Direction of sine wave
                "poly_coeffs": {
                    'x': [0, random.uniform(-1.2, 1.2), 0],
                    'y': [0, random.uniform(-1.2, 1.2), 0],
                    'poly_scale': random.uniform(1.0, 1.5),
                }
            })

        elif path_type == "polynomial":
            params.update({
                "poly_coeffs": {
                    'x': [0, random.uniform(-1.2, 1.2), random.uniform(-0.15, 0.15)],
                    'y': [0, random.uniform(-1.2, 1.2), random.uniform(-0.15, 0.15)],
                    'poly_scale': random.uniform(1.0, 1.5),
                }
            })

        elif path_type == "circle":
            params.update({
                "radius": random.uniform(5, 15),  # Circle radius
                "angular_velocity": random.uniform(0.05, 0.2),  # Speed of rotation
                "center_offset": (random.uniform(-5, 5), random.uniform(-5, 5)),  # Offset from initial position
                "poly_coeffs": {
                    'x': [0, random.uniform(-1.2, 1.2), 0],
                    'y': [0, random.uniform(-1.2, 1.2), 0],
                    'poly_scale': random.uniform(1.0, 1.5),
                }
            })

        elif path_type == "spiral":
            params.update({
                "growth_rate": random.uniform(0.1, 0.3),  # Controls how quickly spiral expands
                "angular_velocity": random.uniform(0.05, 0.2),  # Speed of rotation
                "direction": random.choice(["outward", "inward"]),  # Spiral direction
                "poly_coeffs": {
                    'x': [0, random.uniform(-1.2, 1.2), random.uniform(-0.15, 0.15)],
                    'y': [0, random.uniform(-1.2, 1.2), random.uniform(-0.15, 0.15)],
                    'poly_scale': random.uniform(1.0, 1.5),
                }
            })

        elif path_type == "figure8":
            params.update({
                "scale_x": random.uniform(5, 15),  # Horizontal scale
                "scale_y": random.uniform(5, 15),  # Vertical scale
                "angular_velocity": random.uniform(0.05, 0.2),  # Speed of movement
                "rotation": random.uniform(0, math.pi/2)  # Rotation angle of the figure-8
            })

        elif path_type == "elliptical":
            params.update({
                "semi_major": random.uniform(8, 20),  # Semi-major axis
                "semi_minor": random.uniform(4, 12),  # Semi-minor axis
                "angular_velocity": random.uniform(0.05, 0.2),  # Speed of rotation
                "rotation": random.uniform(0, math.pi/2),  # Rotation of ellipse
                "center_offset": (random.uniform(-5, 5), random.uniform(-5, 5))  # Offset from initial position
            })

        elif path_type == "cubic":
            params.update({
                "coefficients": {
                    'a': random.uniform(-0.001, 0.001),  # x³ coefficient
                    'b': random.uniform(-0.05, 0.05),    # x² coefficient
                    'c': random.uniform(-1.0, 1.0),      # x coefficient
                    'd': 0  # Constant term, usually 0 to start at origin
                },
                "direction": random.choice(["x", "y", "both"]),  # Which coordinate uses cubic function
                "scale": random.uniform(0.5, 2.0)  # Scale factor for the motion
            })

        elif path_type == "zigzag":
            params.update({
                "segment_length": random.uniform(5, 20),  # Length of each segment
                "angle": random.uniform(math.pi/6, math.pi/3),  # Angle of zigzag
                "direction": random.choice(["horizontal", "vertical", "diagonal"]),  # Primary direction
                "randomness": random.uniform(0, 0.3)  # Random variation in segment length/angle
            })

        elif path_type == "exponential":
            params.update({
                "base": random.uniform(1.01, 1.1),  # Base of exponential (slightly above 1)
                "scale": random.uniform(0.5, 2.0),  # Scale factor for the curve
                "direction": random.choice(["up", "down", "left", "right"]),  # Direction of growth
                "decay_factor": random.uniform(0.9, 0.99)  # Optional decay to limit growth
            })

        elif path_type == "hyperbolic":
            params.update({
                "scale_x": random.uniform(5, 15),  # Horizontal scale
                "scale_y": random.uniform(5, 15),  # Vertical scale
                "shift": random.uniform(0.5, 5.0),  # Shift from origin to avoid singularity
                "orientation": random.choice(["x", "y"]),  # Primary axis
                "branch": random.choice(["positive", "negative", "both"])  # Which branch(es) to use
            })

        return params

    raise NotImplementedError(f"Trajectory type {type(trajectory)} is not supported for saving parameters.")


def get_path_type(mnist_label):
    path_types = {
        0: "sine",         # Sinusoidal wave motion: y(t) = a * sin(ωt)
        1: "polynomial",   # Polynomial trajectory: y(t) = ax² + bx + c
        2: "circle",       # Circular orbit: x(t) = r * cos(t), y(t) = r * sin(t)
        3: "spiral",       # Outward spiral: x(t) = at * cos(t), y(t) = at * sin(t)
        4: "figure8",      # Figure-8 curve: x(t) = a * sin(2t), y(t) = a * sin(t)
        5: "elliptical",   # Elliptical path: x(t) = a * cos(t), y(t) = b * sin(t), where a≠b
        6: "cubic",        # Cubic function: y(t) = at³ + bt² + ct + d
        7: "zigzag",       # Zigzag pattern: y(t) = a * abs(((t/b) % 2) - 1)
        8: "exponential",  # Exponential curve: y(t) = a * e^(bt)
        9: "hyperbolic",   # Hyperbolic path: x(t) = a * cosh(t), y(t) = b * sinh(t)
    }

    if mnist_label not in path_types:
        raise ValueError(f"Invalid MNIST label: {mnist_label}. Expected 0-9.")

    return path_types[mnist_label]
