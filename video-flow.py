import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks import get_model
from utils import seed_all, lerp, size_of_model, loss_logging
from utils import compute_average_dict
from datasets import get_dataset
from inference import unproject, t, sample, condiff, joint_ode

from accelerate import Accelerator
from accelerate import DataLoaderConfiguration

from tqdm import tqdm
from omegaconf import OmegaConf

from datetime import datetime
from pathlib import Path
from functools import partial
from einops import rearrange


def main():
    # === load config ===
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

    # === accelerator config ===
    # enable gradient accumulation to fit larger batchsize
    accumulation_steps = cfg.training.accumulation_steps

    # Set split_batches to True to specify actual batch size,
    # otherwise the actual batch size is bs * num_processes (num of GPUs)
    dataloader_config = DataLoaderConfiguration(split_batches=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=accumulation_steps,
        dataloader_config=dataloader_config,
    )
    num_processes = accelerator.num_processes

    # print the config to the console
    accelerator.print(OmegaConf.to_yaml(cfg))

    # === checkpoint code ===
    if accelerator.is_main_process:
        ws_path = Path(cfg.log.exp_path) / (
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            if not cfg.log.comment
            else cfg.log.comment
        )
        writer = SummaryWriter(log_dir=ws_path)
        OmegaConf.save(cfg, ws_path / "config.yaml")

    # === branch ===
    device = accelerator.device

    if cfg.branch.name == "video_flow":
        name = "flow"
    elif cfg.branch.name == "video_biflow":
        name = "biflow"
    elif cfg.branch.name == "condiff":
        name = "condiff"
    else:
        raise ValueError("Invalid branch name")

    # === setup ===
    # Set seed before initializing model.
    seed_all(cfg.seed)
    # NOTE: cfg.training.batchsize is the actual batchsize
    # across all GPUs and accumulation steps

    # This is the total batchsize across all GPUs per step
    global_batchsize = cfg.training.batchsize // accumulation_steps

    # This is the batchsize per GPU per step
    batchsize = global_batchsize // num_processes
    n_iter = cfg.training.n_iter
    cp_iter = n_iter // cfg.log.n_checkpoints
    global_batchsize_inf = cfg.inference.batchsize
    batchsize_inf = global_batchsize_inf // num_processes

    assert (
        batchsize_inf <= batchsize
    ), "inference batchsize must be <= training batchsize"

    inf_noise_level = cfg.inference.noise_level
    n_inf_frames = cfg.inference.n_frames

    # === load data ===
    dataset = get_dataset(cfg.data)
    # NOTE: accelerator will take care of splitting the batchsize across GPUs
    dataloader = DataLoader(
        dataset,
        batch_size=global_batchsize,
        num_workers=12,
        pin_memory=True,
    )

    # === load model ===
    model = get_model(cfg.model)
    accelerator.print(f"The size of model is {size_of_model(model)}")

    # === load optimizer ===
    # Set lr as tensor: https://dev-discuss.pytorch.org/t/compiled-optimizer-w-lr-scheduler-now-supported/2107
    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(cfg.training.lr))

    # === accelerator ===
    # Set seed for the rest of the code
    seed_all(cfg.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if cfg.checkpoints_path:
        accelerator.load_state(cfg.checkpoints_path)
        accelerator.print(f"Loaded checkpoints {cfg.checkpoints_path}")

    # === training function ===
    def train_flow(model, data):
        t = torch.rand(batchsize, device=device)
        x0, x1 = data[:, 0], data[:, 1]

        x_t = lerp(x0, x1, t.view(-1, 1, 1, 1))

        v = model(x_t, t)
        loss = torch.mean((v - (x1 - x0)) ** 2)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        return model, {"loss": loss, "loss_v": loss}

    def train_biflow(model, data):
        t = torch.rand(batchsize, device=device)
        alpha = torch.rand(batchsize, device=device)

        x0, x1 = data[:, 0], data[:, 1]
        z = torch.randn_like(x0, device=device)

        x_t_alpha = lerp(x0, x1, t.view(-1, 1, 1, 1)) + alpha.view(-1, 1, 1, 1) * z

        output = model(x_t_alpha, t, alpha)
        v, d = torch.chunk(output, 2, dim=1)

        loss_v = torch.mean((v - (x1 - x0)) ** 2)
        loss_d = torch.mean((d - z) ** 2)

        loss = loss_v + loss_d

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        return model, {
            "loss": loss,
            "loss_v": loss_v,
            "loss_d": loss_d,
        }

    def train_condiff(model, data):
        alpha = torch.rand(batchsize, device=device)
        x0, x1 = data[:, 0], data[:, 1]

        noise = torch.randn_like(x1, device=device)
        x_alpha = lerp(noise, x1, alpha.view(-1, 1, 1, 1))

        d = model(torch.cat([x_alpha, x0], dim=1), alpha)
        loss = torch.mean((d - (x1 - noise)) ** 2)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        return model, {"loss": loss, "loss_d": loss}

    # === inference function ===
    sample_func = partial(sample, solver=cfg.inference.solver)

    # === training loop ===
    data_iterator = iter(dataloader)
    for it in (
        pbar := tqdm(
            range(n_iter),
            "Training",
            dynamic_ncols=True,
            # Only show progress bar on main process
            disable=not accelerator.is_local_main_process,
        )
    ):
        loss_dict_list = []

        # Accumulate gradients
        for _ in range(accumulation_steps):
            data = next(data_iterator)

            if name == "flow":
                _train = train_flow
            elif name == "biflow":
                _train = train_biflow
            elif name == "condiff":
                _train = train_condiff

            with accelerator.accumulate(model):
                model, loss_dict = _train(model, data)
            loss_dict_list.append(loss_dict)

        # Compute loss stats across accumulation steps
        loss_dict = compute_average_dict(loss_dict_list)

        # Compute loss stats across GPUs
        loss_dict = accelerator.gather(loss_dict)

        # Log loss stats if on main process
        if accelerator.is_main_process:
            loss_dict = {k: v.mean().item() for k, v in loss_dict.items()}
            writer.add_scalar("loss", loss_dict["loss"], it)
            pbar.set_postfix({"loss": f"{loss_dict['loss']:.6f}"})

            loss_logging(writer, loss_dict, it)

        # === checkpointing ===
        if it == 0 or (it + 1) % cp_iter == 0:

            # Save states
            if accelerator.is_main_process:
                accelerator.save_state(ws_path / "checkpoints")

                # save data frames
                len_of_vis = min(8, len(data))
                data_vis = data[:len_of_vis]
                writer.add_images(
                    "data",
                    unproject(rearrange(data_vis, "b t c h w -> (b t) c h w")),
                    it,
                )

            spatial_size = cfg.inference.image_size

            # inference for checkpointing
            x0 = torch.randn((batchsize_inf, 3, *spatial_size), device=device)

            xs_list = []
            stats_list = []

            # some ode's and timepoints
            t_01 = t(batchsize_inf, device)

            # sample first frame
            data_test = next(data_iterator)
            x1 = data_test[:batchsize_inf, 1]

            xs_list.append(x1)

            for i in range(n_inf_frames - 1):
                # For different models
                # set up different initial values and predictors
                x0 = xs_list[-1]
                if name == "flow":
                    initial_value = x0
                    # not used, just for matching the random seeds
                    _ = torch.randn_like(x0, device=device)
                    predictor = model
                elif name == "biflow":
                    initial_value = (
                        x0 + torch.randn_like(x0, device=device) * inf_noise_level
                    )
                    start = (0, inf_noise_level)
                    end = (1, 0)
                    predictor = joint_ode(model, start, end)
                elif name == "condiff":
                    initial_value = torch.randn_like(x0, device=device)
                    predictor = condiff(model, x0)

                # Solve by the specified ODE solver
                stats, xs = sample_func(predictor, initial_value, t_01)
                stats_list.append(stats["n_f_evals"].to(xs))
                x1 = xs[:, 1, ...]

                x1 = x1.clamp(-1.0, 1.0)
                xs_list.append(x1)

            xs = torch.stack(xs_list, dim=1)  # B x T x C x H x W
            stats = torch.stack(stats_list, dim=-1)  # B x T

            # gather inference across all GPUs
            xs = accelerator.gather(xs)
            stats = accelerator.gather(stats)

            xs_ff = xs[:, 0]

            if accelerator.is_main_process:
                writer.add_scalar("checkpoints/n_f_evals", stats.mean(), it)

                writer.add_images("checkpoints/first_frame", unproject(xs_ff), it)
                writer.add_video("checkpoints/inference", unproject(xs), it, fps=10)

    if accelerator.is_main_process:
        writer.close()


if __name__ == "__main__":
    main()
