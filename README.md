# Detection Moving MNIST Dataset

This dataset extends the [original Moving MNIST dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/). A few
variations on how digits move are added.

In this dataset, each frame is padded to have a resolution of image size 128x128. Each frame is also provided with annotations for object detection (center point detection).

## How to generate torch-tensor-format datasets

```text
detection-moving-mnist % python3 generate.py -h                                                                                         
usage: generate.py [-h] [--version VERSION] [--split SPLIT] [--num_frames_per_video NUM_FRAMES_PER_VIDEO] [--num_videos NUM_VIDEOS] [--whole_dataset] [--seed SEED]

Generate Detection MovingMNIST dataset with specified parameters.

options:
  -h, --help            show this help message and exit
  --version VERSION     MMNIST version: easy
  --split SPLIT         Dataset splits: train, test
  --num_frames_per_video NUM_FRAMES_PER_VIDEO
                        Number of frames per video.
  --num_videos NUM_VIDEOS
                        Number of videos.
  --whole_dataset       We make sure all MNIST digits are used for the dataset.
  --seed SEED           Seed.
```

Example:
```shell
python3 generate.py --split train --version easy --num_frames_per_video 20 --num_videos 60000
```

## How to convert torch-tensor-format to huggingface videofolder format

```text
python3 to_video.py -h                                                                                         
usage: to_video.py [-h] [--version VERSION] [--split SPLIT] [--in_place]

Convert torch-tensor-format to huggingface videofolder format.

options:
  -h, --help         show this help message and exit
  --version VERSION  MMNIST version: easy, medium, hard, random
  --split SPLIT      Dataset splits: train, test
  --in_place         Remove source files during conversion to save space
```

Example:
```shell
python3 to_video.py --version easy --split test
```

Video conversion uses a rate of 10 frames per second. This can be adjusted in `src/utils/utils.py`.

## How to calculate dataset statistics (huggingface videofolder format).

Important this script supports only huggingface videofolder format.

```text
python3 calculate_dataset_statistics.py -h
usage: calculate_dataset_statistics.py [-h] --dataset_dir DATASET_DIR [--splits SPLITS [SPLITS ...]]

Calculate dataset statistics and create distribution histograms

options:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        Root directory of dataset in Hugging Face videofolder format
  --splits SPLITS [SPLITS ...]
                        Dataset splits to process (e.g., train,test)

```

Example:
```shell
python3 calculate_dataset_statistics.py --dataset_dir mmnist-dataset/huggingface-videofolder-format/mmnist-easy
```

## Acknowledgements

This project is based on and modified from the repository:

* [captioned-moving-mnist](https://github.com/YichengShen/captioned-moving-mnist/tree/main)

We extend our gratitude to the original author @YichengShen for their work.
