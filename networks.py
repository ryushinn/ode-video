from diffusers import UNet2DModel
import torch
from pathlib import Path
from safetensors.torch import load_model


# Wrap UNet2DModel to return only the sample
class MyUNet2DModel(UNet2DModel):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).sample


class TwoModel(torch.nn.Module):

    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, t, alpha=None):
        return torch.concat([self.model1(x, t, alpha), self.model2(x, t, alpha)], dim=1)


def get_default_s(in_channels, out_channels, c, cond=False):
    block_out_channels = (c, c, 2 * c, 2 * c, 4 * c, 4 * c)
    down_block_types = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    return MyUNet2DModel(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
        class_embed_type="timestep" if cond else None,
    )


def get_default(in_channels, out_channels, c, cond=False):
    block_out_channels = (c, c, 2 * c, 2 * c, 2 * c, 4 * c, 4 * c)
    down_block_types = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    return MyUNet2DModel(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
        class_embed_type="timestep" if cond else None,
    )


def get_model(model_config):
    config: str = model_config.config
    class_cond = "cond" in config
    digits = list(int(part) for part in config.split("_") if part.isdigit())
    in_channels, out_channels, c = digits

    _get_model = get_default_s if "small" in config else get_default

    if "two_model" in config:
        model1 = _get_model(in_channels, out_channels, c, class_cond)
        model2 = _get_model(in_channels, out_channels, c, class_cond)
        model = TwoModel(model1, model2)
    else:
        model = _get_model(in_channels, out_channels, c, class_cond)

    if model_config.weights_path:
        print(f"Loading model from {model_config.weights_path}")
        weights_path = Path(model_config.weights_path)
        load_model(model, weights_path)

    return model
