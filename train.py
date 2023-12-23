from config import Config
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.plot import *
from model.model import DFUNet
from utils.misc import seed_all
from dataset import MINISTData
from model.module import DDPMScheduler
from model.inf import inference
from utils.misc import date


# Config
config = Config()
save_dir = f'demo/epoch-{config.epochs}/' + date()
# set seed
seed_all(config.seed)
# Dataset and dataloader
train_dataset = MINISTData(config, 'data', return_label=True)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
# scheduler
scheduler = DDPMScheduler(config)
# Model
model = DFUNet(config).to(config.device)
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

# train
for epoch in range(config.epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    model.train()
    for image, _ in train_dataloader:
        batch = image.shape[0]
        timesteps = scheduler.sample_timesteps(batch)
        noise = torch.randn(image.shape).to(config.device)
        noisy_image = scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)
        # compute loss
        pred = model(noisy_image, timesteps)
        loss = torch.nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        # clip grad
        orig_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        progress_bar.update(1)
        logs = {'loss': loss.detach().item(), 'epoch': epoch+1, 'orig_grad': orig_grad}
        progress_bar.set_postfix(**logs)
    
    model.eval()
    image = inference(scheduler, config.num_inf_images, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    plot_images(image, save_dir=save_dir)
    print('Max_Cudamemory_Allocated:{}'.format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))


# save checkpoints
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, 'checkpoint/' + date() + str(config.epochs))
