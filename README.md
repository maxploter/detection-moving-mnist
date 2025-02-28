# Detection Moving MNIST Dataset

This dataset extends the [original Moving MNIST dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/). A few
variations on how digits move are added.

In this dataset, each frame is padded to have a resolution of image size 128x128. Each frame is also provided with annotations for object detection (center point detection).

## How to generate torch-tensor-format datasets

```text
python3 generate.py -h
usage: generate.py [-h] [--version VERSION] [--split SPLIT] [--num_frames_per_video NUM_FRAMES_PER_VIDEO] [--num_videos NUM_VIDEOS]

Generate Detection MovingMNIST dataset with specified parameters.

options:
  -h, --help            show this help message and exit
  --version VERSION     MMNIST version: easy
  --split SPLIT         Dataset splits: train, test
  --num_frames_per_video NUM_FRAMES_PER_VIDEO
                        Number of frames per video.
  --num_videos NUM_VIDEOS
                        Number of videos.
```

Example:
```shell
python3 generate.py --split train --version easy --num_frames_per_video 2 --num_batches 2 --batch_size 1
```

## How to convert torch-tensor-format to huggingface videofolder format

```text
python3 to_video.py -h
usage: to_video.py [-h] [--version VERSION] [--split SPLIT]

Convert torch-tensor-format to huggingface videofolder format.

options:
  -h, --help         show this help message and exit
  --version VERSION  MMNIST version: easy, medium, hard, random
  --split SPLIT      Dataset splits: train, test
```

Example:
```shell
python3 to_video.py --version easy --split test
```

Video conversion uses a rate of 10 frames per second. This can be adjusted in `src/utils/utils.py`.

## Dataset Versions

This dataset currently supports 4 versions: easy, medium, hard, and random.

Suggestions:

- Use a small `[num_frames_per_video]`, such as 20, for the easy version because the digit will quickly move out of
  bounds.

## Acknowledgements

This project is based on and modified from the repository:

* [captioned-moving-mnist](https://github.com/YichengShen/captioned-moving-mnist/tree/main)

We extend our gratitude to the original author @YichengShen for their work.
