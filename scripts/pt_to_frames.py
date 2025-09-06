import torch
import os
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

def extract_frames_from_videos(root_folder, output_root_folder):
    pt_extensions = ('.pt', '.pth')
    
    for subdir, _, files in os.walk(root_folder):
        print(f"Processing {subdir}")
        for file in (pbar:= tqdm(files)):
            if file.endswith(pt_extensions):
                pt_path = os.path.join(subdir, file)
                pt_name = os.path.splitext(file)[0]
                
                # Create the corresponding output directory structure
                relative_subdir = os.path.relpath(subdir, root_folder)
                output_folder = os.path.join(output_root_folder, relative_subdir, pt_name)
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                video = torch.load(pt_path)
                num_frames = video.shape[0]
                num_digits = len(str(num_frames))
                
                for i, frame in enumerate(video):
                    frame_image = Image.fromarray(frame.numpy())
                    frame_image.save(os.path.join(output_folder, f"{i:0{num_digits}d}.png"))

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    extract_frames_from_videos(cfg.root_folder, cfg.output_root_folder)