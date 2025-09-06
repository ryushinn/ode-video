import torch
from torch.utils.data import DataLoader

from networks import get_model
from utils import seed_all, size_of_model
from datasets import get_dataset
from inference import unproject, t, sample, condiff, joint_ode
import torchvision.io as tvio

from tqdm import tqdm
from omegaconf import OmegaConf

from datetime import datetime
from pathlib import Path
from functools import partial
from einops import rearrange
from PIL import Image
import numpy as np
import threading


def main():
    # === load config ===
    torch.set_float32_matmul_precision("high")
    cli_config = OmegaConf.from_cli()
    if hasattr(cli_config, "yaml"):
        yaml = Path(cli_config.yaml)
        del cli_config.yaml
    else:
        yaml = Path("config/video-flow.yaml")
    cfg = OmegaConf.load(yaml)
    # set to True to disallow adding unknown fields to the config
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, cli_config)

    # === checkpoint code ===
    ws_path = Path(cfg.log.exp_path) / (
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if not cfg.log.comment
        else cfg.log.comment
    )
    ws_path.mkdir(parents=True, exist_ok=True)

    # print the config to the console
    print(OmegaConf.to_yaml(cfg))

    # === setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg.seed)
    n_samples = cfg.inference.n_samples
    batchsize = cfg.inference.batchsize
    assert n_samples % batchsize == 0
    inf_noise_level = cfg.inference.noise_level
    n_inf_frames = cfg.inference.n_frames
    solve_backward = cfg.inference.backward
    if solve_backward:
        assert (
            cfg.branch.name == "video_biflow"
        ), "Only video biflow supports backward sampling"

    # === load data ===
    dataset_test = get_dataset(cfg.data)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batchsize,
        num_workers=4,
        pin_memory=True,
    )

    # === load model ===
    model = get_model(cfg.model)
    model = model.to(device)
    model.requires_grad_(False)
    model = model.eval()
    print(f"The size of model is {size_of_model(model)}")

    # seed is set for the rest of the code
    seed_all(cfg.seed)

    # === inference function ===
    sample_func = partial(sample, solver=cfg.inference.solver)

    # === training loop ===
    data_test_iterator = iter(dataloader_test)
    # generate n_samples samples, but each time only with batchsize samples at most
    stats_final_list = []
    threads = []
    for batch_idx in (
        pbar := tqdm(
            range(0, n_samples, batchsize),
            "Inference",
            dynamic_ncols=True,
        )
    ):
        # inference for checkpointing
        xs_list = []
        stats_list = []

        # some ode's and timepoints
        t_01 = t(batchsize, device)

        # sample first frame
        data_test = next(data_test_iterator)
        x0 = data_test[:, 1]
        x0 = x0.to(device)

        xs_list.append(x0)

        for i in range(n_inf_frames - 1):
            x0 = xs_list[-1]
            if cfg.branch.name == "video_flow":
                initial_value = x0
                # not used, just for matching the random seeds
                _ = torch.randn_like(x0, device=device)
                predictor = model
            elif cfg.branch.name == "video_biflow":
                initial_value = x0 + torch.randn_like(
                    x0, device=device
                ) * inf_noise_level
                if solve_backward:
                    start = (0, 0)
                    end = (1, inf_noise_level)
                else:
                    start = (0, inf_noise_level)
                    end = (1, 0)
                predictor = joint_ode(model, start, end)
            elif cfg.branch.name == "condiff":
                initial_value = torch.randn_like(x0, device=device)
                predictor = condiff(model, x0)
            stats, xs = sample_func(
                predictor, initial_value, t_01 if not solve_backward else 1 - t_01
            )
            stats_list.append(stats["n_f_evals"].to(xs))
            x1 = xs[:, 1, ...]

            x1 = x1.clamp(-1.0, 1.0)
            xs_list.append(x1)

        xs = torch.stack(xs_list, dim=1)  # B x T x C x H x W
        xs = unproject(xs)
        stats = torch.stack(stats_list, dim=-1)  # B x T
        stats_final_list.append(stats)

        def save_images(batch_idx, xs, ws_path):
            for i, frames in enumerate(xs):
                frames_PIL: list[Image.Image] = []
                for j, frame in enumerate(frames):
                    save_path = ws_path / f"{i+batch_idx:05d}" / f"{j:05d}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    frame = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    frame_PIL = Image.fromarray(frame)
                    frame_PIL.save(save_path)
                    frames_PIL.append(frame_PIL)

                # Multimedia
                frames_PIL[0].save(
                    ws_path / f"{i+batch_idx:05d}" / f"{0:05d}.gif",
                    save_all=True,
                    append_images=frames_PIL[1:],
                    duration=100,  # fps = 10
                    loop=0,
                )

                tvio.write_video(
                    ws_path / f"{i+batch_idx:05d}" / f"{0:05d}.mp4",
                    (
                        rearrange(frames, "T C H W -> T H W C").cpu().numpy() * 255
                    ).astype(np.uint8),
                    fps=10,
                )

        save_thread = threading.Thread(
            target=save_images, args=(batch_idx, xs, ws_path)
        )
        save_thread.start()
        threads.append(save_thread)

        pbar.set_postfix({"n_f_evals": stats.mean(dtype=torch.float32).item()})

    stats_final = torch.concatenate(stats_final_list, dim=0)
    with open(ws_path / f"stats.pt", "wb") as f:
        torch.save(stats_final.to("cpu"), f)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
