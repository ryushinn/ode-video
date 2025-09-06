import random
from pathlib import Path
import shutil
from omegaconf import OmegaConf

def split_dataset(source_dir, output_dir, train_ratio=0.8, random_seed=42):
    # Set a random seed for reproducibility
    random.seed(random_seed)

    # Define the directories using pathlib.Path
    source_dir = Path(source_dir)
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"

    # Create train and test directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Function to determine if a directory is a leaf directory (i.e., no subdirectories)
    def is_leaf_dir(d: Path) -> bool:
        return not any(child.is_dir() for child in d.iterdir())

    # Recursively find all leaf directories under source_dir
    clip_folders = [d for d in source_dir.rglob("*") if d.is_dir() and is_leaf_dir(d)]

    # Shuffle the list of clip folders
    random.shuffle(clip_folders)

    # Compute the split index for training split
    split_index = int(len(clip_folders) * train_ratio)
    train_folders = clip_folders[:split_index]
    test_folders = clip_folders[split_index:]

    # Copy the clip folders into the respective train and test directories.
    # Here we preserve the relative folder structure from source_dir.
    for folder in train_folders:
        relative_path = folder.relative_to(source_dir)
        destination = train_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, destination)

    for folder in test_folders:
        relative_path = folder.relative_to(source_dir)
        destination = test_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, destination)

    print(f"Total clip folders: {len(clip_folders)}")
    print(f"Training clip folders: {len(train_folders)}")
    print(f"Testing clip folders: {len(test_folders)}")

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    split_dataset(
        source_dir=cfg.source_dir,
        output_dir=cfg.output_dir,
        train_ratio=cfg.get('train_ratio', 0.8),
        random_seed=cfg.get('random_seed', 42)
    )
