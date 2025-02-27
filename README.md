# Detection Moving MNIST Dataset

This dataset extends the [original Moving MNIST dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/). A few
variations on how digits move are added.

In this dataset, each frame is padded to have a resolution of image size parameter. Each frame is also provided with annotations for object detection (center point detection).

## How to generate torch-tensor-format datasets

```shell
python3 generate.py [version] [num_frames_per_video] [num_batches] [batch_size]
```

## How to convert torch-tensor-format to video-format

```shell
python3 to_video.py [version]
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

We extend our gratitude to the original authors for their work.
