# Generative Video Bi-flow

> by [Chen Liu](https://ryushinn.github.io/) and [Tobias Ritschel](https://www.homepages.ucl.ac.uk/~ucactri/)
>
> _International Conference on Computer Vision (ICCV 2025)_
>
> Please also check out our ([Paper](https://arxiv.org/abs/2503.06364) | [Project Page](https://ryushinn.github.io/ode-video))

This repo provides the official implementation of our paper in PyTorch.

## Setup

### Install

```bash
# 1. Clone the repo
git clone https://github.com/ryushinn/ode-video.git
cd ode-video
# 2. Recommend installing in a new virtual env with python 3.10, such as conda:
conda create -n ode-video python=3.10
conda activate ode-video
# 3. Install the dependencies
pip install -r requirements.txt
```

Our test environment is Ubuntu 22.04.4 x64 and NVIDIA RTX4090 GPU with CUDA 12.

### Data

Our dataloader expects the following folder structure:

```bash
data
└── {Dataset}
    ├── {train_split}
    │   └── ... # Nested folders are allowed
    │       └── {clip_folder}
    │           ├── 000001.jpg # first frame
    │           ├── 000002.jpg # second frame
    │           └── ...
    └── {test_split}
        └── ...
            └── {clip_folder}
                ├── 000001.jpg
                ├── 000002.jpg
                └── ...
```

Every (sub)folder in train or test split should only contain consecutive frames from the same video clip, which are named in sorted order.

For example, you can setup `sky` dataset as in the above format, using:

```bash
# If daily download limit was reached, please download manually at
# https://drive.google.com/uc?id=1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo
gdown 1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo -O sky_timelapse.zip
unzip sky_timelapse.zip -d data
rm sky_timelapse.zip
```

For those datasets at a different format other than frames, you can use `scripts/pt_to_frames.py` (e.g., [`CARLA`](https://github.com/plai-group/flexible-video-diffusion-modeling?tab=readme-ov-file#preparing-data)) or `scripts/video_to_frames.py` (e.g., [`minerl`](https://archive.org/details/minerl_navigate) and [`mazes`](https://archive.org/details/gqn_mazes)) to convert them to image frames.

If the dataset does not come with a default train-test split, you can use `scripts/split.py` to setup one, e.g., for [`biking`](https://github.com/NVlabs/long-video-gan?tab=readme-ov-file#preparing-datasets) and [`riding`](https://github.com/NVlabs/long-video-gan?tab=readme-ov-file#preparing-datasets).

## Usages

### Pre-trained weights

You can download the [pre-trained weights](https://drive.google.com/file/d/1SOylrO6udRW_Qd6YRRIXHnv3FmHc3ukL/view?usp=sharing) for six datasets we report in our paper.

```bash
# If daily download limit was reached, please download manually
gdown 1SOylrO6udRW_Qd6YRRIXHnv3FmHc3ukL -O checkpoints_ode-video.zip
unzip checkpoints_ode-video.zip
rm checkpoints_ode-video.zip
```

### Training (from scratch)

```bash
# USAGE:
#   train_videoflow.sh <data_path> <exp_path> <image_size> <num_processes> <accumulation_steps>
# ARGS:
#   <data_path>          : the folder of your training dataset
#   <exp_path>           : the folder to save checkpoints and logs
#   <image_size>         : resize the training images to this size
#   <num_processes>      : the number of GPUs
#   <accumulation_steps> : the number of steps you accumulate the gradients from several batches.
#       This will NOT affect the actual batch size,
#       but allow you to use a large batch size in limited GPU memory
#       by performing one optimizer step after several backward passes

bash scripts/train_videoflow.sh data/sky_timelapse/sky_train experiments_weights/sky 128 1 8
```

Above is an example to train `condiff`, `flow`, and `bi-flow` for the dataset `sky`.
If out of GPU memory, you can use more accumulation steps.

### Sampling

> Note that our trained ODEs can generate next frames but the first frame has to be given or generated separately.
> Thus you would need to setup the test split of the corresponding dataset to sample (generate) videos using the trained weights.

To sample the trained models, you can use:

```bash
# USAGE:
#   sample_videoflow.sh <data_path> <model_path> <exp_path> <image_size> <n_samples> <batchsize> <n_frames>
# ARGS:
#   <data_path>  : the folder of your test dataset
#   <model_path> : the folder of your training checkpoints and logs
#   <exp_path>   : the folder to save sampling results
#   <image_size> : the image size you sample
#   <n_samples>  : the number of videos generated, must be a multiple of the batch size
#   <batchsize>  : the batch size in sampling
#   <n_frames>   : the number of frames in each sample

bash scripts/sample_videoflow.sh data/sky_timelapse/sky_test experiments_weights/sky experiments_inference/sky 128 8 1 32
```

The above command samples 8 videos, each of which has 32 frames. The sampling script will sample `condiff` and `flow`, together with `bi-flow` under four different levels of inference noises.

## Citation

```bibtex
@article{liu2025generative,
  title={Generative Video Bi-flow},
  author={Liu, Chen and Ritschel, Tobias},
  journal={arXiv preprint arXiv:2503.06364},
  year={2025}
}
```
