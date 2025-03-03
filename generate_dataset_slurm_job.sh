#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-mmnist-dataset"
#SBATCH --time=5:00:00
#SBATCH --output=not_tracked_dir/slurm/%j_slurm_%x.out
#SBATCH --partition=main
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load miniconda3

# Make environment creation optional if not exists
if ! conda info --envs | grep -q 'detection_moving_mnist'; then
    conda create -n detection_moving_mnist python=3.11 -y
    conda activate detection_moving_mnist
    # Install requirements only if environment was created
    pip install -r requirements.txt
else
    conda activate detection_moving_mnist
fi

# Create directory optionally if not exists
mkdir -p not_tracked_dir/slurm

split = 'train'
python3 generate.py --split {split} --version easy --num_frames_per_video 20 --num_videos 60000 --num_videos_hard 120000 --hf_arrow_format --whole_dataset

split = 'test'
python3 generate.py --split {split} --version easy --num_frames_per_video 20 --num_videos 10000 --num_videos_hard 20000 --hf_arrow_format --whole_dataset

python3 calculate_dataset_statistics.py --dataset_dir mmnist-dataset/huggingface-arrow-format/mmnist-easy