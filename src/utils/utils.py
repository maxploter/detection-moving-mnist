import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


def load_dataset(directory):
    frames = []
    captions = []

    frame_files = sorted([f for f in os.listdir(directory) if f.endswith("_frames.pt")])
    caption_files = sorted(
        [f for f in os.listdir(directory) if f.endswith("_captions.txt")]
    )

    for frame_file, caption_file in zip(frame_files, caption_files):
        # Load frames
        batch_frames = torch.load(os.path.join(directory, frame_file))
        frames.append(batch_frames)

        # Load captions
        batch_captions = []
        with open(os.path.join(directory, caption_file), "r") as file:
            batch_caption_list = []
            for line in file:
                if line.strip():
                    batch_caption_list.append(line.strip())
                else:
                    if batch_caption_list:
                        batch_captions.append(batch_caption_list)
                        batch_caption_list = []
        captions.append(batch_captions)

    return frames, captions


def create_video_from_frames(
    frames,
    output_filename,
    frame_rate=10.0,
    resolution=(128, 72),
    colormap=cv2.COLORMAP_BONE,
):
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_filename, fourcc, frame_rate, resolution)

    for frame in frames:
        frame = frame.detach()
        frame_pil = TF.to_pil_image(frame)
        frame_np = np.array(frame_pil)
        color_image = cv2.applyColorMap(frame_np, colormap)
        video_writer.write(color_image)

    video_writer.release()
