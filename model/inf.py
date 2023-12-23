from typing import Optional
import torch
from config import Config

def inference(model, scheduler, images: int, config: Config, noise: Optional[torch.Tensor] = None):
    if noise == None:
        noisy_sample = torch.randn((images, config.input_channels, config.image_size, config.image_size)).to(config.device)
    else:
        noisy_sample = noise.to(config.device)

    for t in scheduler.inf_timesteps:
        with torch.no_grad():
            noisy_pred = model(noisy_sample, t[None].to(config.device)).sample
            noisy_sample = scheduler.step(noisy_pred, t, noisy_sample)
            
    return noisy_sample
