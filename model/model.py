from diffusers import UNet2DModel
from config import Config
import torch


class DFUNet(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super(DFUNet, self).__init__()
        self.config = config

        self.model = UNet2DModel(
            sample_size=config.image_size, # The size of images
            in_channels=config.input_channels,
            out_channels=config.input_channels,
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                              "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D",  "UpBlock2D", "UpBlock2D",
                            "UpBlock2D", "UpBlock2D", "UpBlock2D")
        )

    def forward(self, x, ts):
        return self.model(x, ts)[0]
