import logging
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
        path="data",
        num_digits=(
            1,
            2,
        ),  # random choice in the tuple to set number of moving digits
        num_frames=10,  # number of frames to generate
        concat=True,  # if we concat the final results (frames, 1, 28, 28) or a list of frames
    ):
        self.mnist = MNIST(path, download=True)
        self.total_data_num = len(self.mnist)
        self.trajectory = trajectory
        self.affine_params = affine_params
        self.num_digits = num_digits
        self.num_frames = num_frames
        self.padding = get_padding(128, 72, 28, 28)  # MNIST images are 28x28
        self.concat = concat

    def random_place(self, img):
        """Randomly place the digit inside the canvas"""
        x = random.randint(-self.padding[0], self.padding[0])
        y = random.randint(-self.padding[1], self.padding[1])
        placed_img = TF.affine(img, translate=[x, y], angle=0, scale=1, shear=[0])
        return placed_img, (x, y)

    def random_digit(self):
        """Get a random MNIST digit randomly placed on the canvas"""
        img, label = self.mnist[random.randrange(0, self.total_data_num)]
        img = TF.to_tensor(img)
        pimg = TF.pad(img, padding=self.padding)
        placed_img, initial_position = self.random_place(pimg)
        return placed_img, initial_position, label

    def _one_moving_digit(self):
        digit, initial_position, label = self.random_digit()
        traj = self.trajectory(
            label,
            self.affine_params,
            n=self.num_frames - 1,
            padding=self.padding,
            initial_position=initial_position,
        )
        frames, captions = traj(digit)
        return torch.stack(frames), captions, label

    def __getitem__(self, i):
        moving_digits, captions_list, labels = zip(
            *(self._one_moving_digit() for _ in range(random.choice(self.num_digits)))
        )
        moving_digits = torch.stack(moving_digits)
        combined_digits = moving_digits.max(dim=0)[0]

        if len(labels) == 1:
            combined_captions = [f"A digit {labels[0]} is on the first frame."]
        else:
            combined_captions = [f"Digits {labels} are on the first frame."]
        for frame_idx in range(len(captions_list[0])):
            frame_caption = "\t".join(
                [captions[frame_idx] for captions in captions_list]
            )
            combined_captions.append(frame_caption)
        return (
            (
                combined_digits
                if self.concat
                else [t.squeeze(dim=0) for t in combined_digits.split(1)]
            ),
            combined_captions,
        )

    def get_batch(self, bs=32):
        batch = [self[0] for _ in range(bs)]
        frames, captions = zip(*batch)
        frames = torch.stack(frames)
        return frames, captions

    def save(self, directory, n_batches, bs):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for batch_index in tqdm(range(n_batches), desc="Processing batches"):
            batch_frames, batch_captions = self.get_batch(bs)
            torch.save(
                batch_frames, os.path.join(directory, f"batch_{batch_index}_frames.pt")
            )

            with open(
                os.path.join(directory, f"batch_{batch_index}_captions.txt"), "w"
            ) as file:
                for captions in batch_captions:
                    for single_frame_caption in captions:
                        file.write(single_frame_caption + "\n")
                    file.write("\n\n")

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


def apply_n_times(tf, x, n):
    """Apply `tf` to `x` `n` times, return all values"""
    sequence = [x]
    captions = []
    for _ in range(n):
        x_new, caption = tf(sequence[-1])
        sequence.append(x_new)
        captions.append(caption)
    return sequence, captions
