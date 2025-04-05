# Detection Moving MNIST Dataset

![Detection Moving MNIST Dataset (Easy) annotated video 0](./annotated_video_0.gif)
![Detection Moving MNIST Dataset (Easy) annotated video 1](./annotated_video_1.gif)
![Detection Moving MNIST Dataset (Easy) annotated video 2](./annotated_video_2.gif)

This dataset extends the [original Moving MNIST dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/). A few
variations on how digits move are added.

In this dataset, each frame is padded to have a resolution of image size 128x128. Each frame is also provided with annotations for object detection (center point detection).

## Setup repository

```shell
conda create -n detection_moving_mnist python=3.11
conda activate detection_moving_mnist
pip install -r requirements.txt
```

## How to generate torch-tensor-format datasets

```text
python3 generate.py -h                    
usage: generate.py [-h] [--version VERSION] [--split SPLIT] [--num_frames_per_video NUM_FRAMES_PER_VIDEO] [--num_videos NUM_VIDEOS] [--num_videos_hard NUM_VIDEOS_HARD] [--whole_dataset] [--seed SEED] [--hf_videofolder_format] [--hf_arrow_format]

Generate Detection MovingMNIST dataset with specified parameters.

options:
  -h, --help            show this help message and exit
  --version VERSION     MMNIST version: easy
  --split SPLIT         Dataset splits: train, test
  --num_frames_per_video NUM_FRAMES_PER_VIDEO
                        Number of frames per video.
  --num_videos NUM_VIDEOS
                        Number of videos.
  --num_videos_hard NUM_VIDEOS_HARD
                        Number of videos hard limit used when whole_dataset is set.
  --whole_dataset       We make sure all MNIST digits are used for the dataset.
  --seed SEED           Seed.
  --hf_videofolder_format
                        Save in Hugging Face video folder format.
  --hf_arrow_format     Save in Hugging Face arrow format.
```

Example:
```shell
python3 generate.py --split train --version easy --num_frames_per_video 20 --num_videos 60000 --num_videos_hard 120000 --whole_dataset --hf_arrow_format
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

## How to calculate dataset statistics (huggingface arrow format).

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
python3 calculate_dataset_statistics.py --dataset_dir mmnist-dataset/huggingface-arrow-format/mmnist-easy
```

## Acknowledgements

This project is based on and modified from the repository:

* [captioned-moving-mnist](https://github.com/YichengShen/captioned-moving-mnist/tree/main)

We extend our gratitude to the original author @YichengShen for their work.
