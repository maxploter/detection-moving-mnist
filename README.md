# Captioned Moving MNIST Dataset

This dataset extends the [original Moving MNIST dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/). A few
variations on how digits move are added.

In this dataset, each frame is padded to have a resolution of 128x72. Each frame is also provided with a text caption.

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

## Rye Commands

- Sync: `rye sync`
- Add dependency: `rye add numpy`
- Remove dependency: `rye remove numpy`
- Activate the virtualenv: `. .venv/bin/activate`
- Deactivate: `deactivate`
