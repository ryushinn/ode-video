from typing import Optional
from pyparsing import Callable
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from pathlib import Path
import torchvision.io as tvio
import os


def is_image_file(file_path):
    # Define a set of common image file extensions
    image_extensions = {".jpg", ".jpeg", ".png"}
    return file_path.suffix.lower() in image_extensions


def build_frame_handles(
    directory: Path,
    n_frames: Optional[int] = None,
    file_filter: Callable = is_image_file,
) -> list[list[Path]]:
    """Build frame handles from a directory.

    Args:
        directory (Path): The root directory to search for image files.
        n_frames (int, optional): The maximum number of frames per clip. Defaults to None, meaning no limit.
        file_filter (callable, optional): A function to filter files. Defaults to is_image_file.

    Returns:
        list[list[Path]]: A list of clips, each containing a list of image file paths.
    """
    clips = []
    for root, dirs, files in os.walk(directory):
        # Sort to impose a consistent order for all machines
        dirs.sort()

        root = Path(root)

        files = [root / file for file in files]
        files = [file for file in files if file_filter(file)]

        try:
            # Sort the files by their integer value
            files.sort(key=lambda x: int(str(x).split("/")[-1].split(".")[0]))
        except ValueError:
            # If the file names are not integers, sort them by their string value
            files.sort()

        if len(files) == 0:
            continue

        if n_frames == None:
            clips.append(files)
        else:
            for i in range(0, len(files), n_frames):
                if i + n_frames > len(files):
                    break
                clip = files[i : i + n_frames]
                clips.append(clip)

    return clips


class FrameSet(IterableDataset):

    def __init__(self, clips, loader):
        self.clips = clips
        self.loader = loader

    def __iter__(self):
        return self

    def __next__(self):
        clip = self.clips[torch.randint(0, len(self.clips), ())]

        random_idx = torch.randint(0, len(clip) - 1, ())
        data = self.loader(clip[random_idx : random_idx + 2])

        return data


def get_loader(image_size, to_float=True, normalize=True):
    my_transforms = []
    my_transforms.append(transforms.Resize(image_size))
    if to_float:
        my_transforms.append(transforms.ConvertImageDtype(torch.float32))
    if normalize:
        my_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    data_trans = transforms.Compose(my_transforms)

    def image_loader(image_paths):
        images = torch.stack(
            [tvio.read_image(image_path) for image_path in image_paths]
        )
        return data_trans(images)

    return image_loader


def get_dataset(data_config):
    image_loader = get_loader(data_config.image_size)

    # NOTE: for sky dataset, the max number of consecutive frames is 32.
    # We set it here for sky dataset.
    if "sky" in str(data_config.path):
        print("Now loading sky dataset")
        data_config.n_frames = 32
        print(f"Setting n_frames to {data_config.n_frames}")

    data_handles = build_frame_handles(data_config.path, data_config.n_frames)
    loader = image_loader
    dataset = FrameSet(data_handles, loader)

    return dataset
