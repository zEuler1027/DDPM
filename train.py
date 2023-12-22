from config import Config
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as v2
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy
from utils.plot import *
from model.model import DFUNet


# Config
config = Config()


# Dataset
class MINISTData(Dataset):
    def __init__(self,
            config: 'Config',
            dataset_dir: str,
            return_label=False
    ) -> None:
        super().__init__()

        self.config = config
        self.target_size = config.image_size
        self.return_label = return_label
        self.transform = v2.Compose([
            v2.Resize((self.target_size, self.target_size), InterpolationMode.BILINEAR),
            v2.ToTensor(), 
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize([0.5], [0.5])
        ])
        self.dataset = MNIST(dataset_dir, train=True, transform=self.transform, download=True)

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx: int):
        image, label = self.dataset.__getitem__(index=idx)
        if self.return_label:
            label = torch.tensor(label)
            return image.to(self.config.device), label.to(self.config.device) # [1, 64, 64], [1, ]
        else:
            return image.to(self.config.device) 
        

# Scheduler
class Scheduler:
    def __init__(self, config: 'Config') -> None:
        self.config = config

        # Timesteps
        self.num_train_timesteps: int = self.config.timesteps # train_steps
        self.num_inf_timesteps: int = self.config.timesteps # inference_steps
        self.set_timesteps()

        # beta and alpha
        self.beta_start: float = self.config.beta0
        self.beta_end: float = self.config.beta1
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32).to(self.config.device)
        self.alphas = 1.0 - self.betas # alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # alpha_bar

    def set_timesteps(self):
        '''
        caculate inf_timesteps -> self.inf_timesteps (torch.Tensor)
        '''
        step_ratio = self.num_train_timesteps // self.num_inf_timesteps
        timesteps = (numpy.arange(0, self.num_inf_timesteps) * step_ratio).round()[::-1].copy().astype(numpy.int64)

        self.inf_timesteps = torch.from_numpy(timesteps).to(self.config.device)
    
    def add_noise(self,
                  image: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        '''
        add noise to x0 (image)
        x_t = √(α_t)x_0 + √(1-α_t) ε
        '''
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps]) # √α_bar_t
        while len(sqrt_alpha_prod.shape) < len(image.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[timesteps])
        while len(sqrt_one_minus_alpha_prod.shape) < len(image.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * image + sqrt_one_minus_alpha_prod * noise

        return noisy_samples.to(self.config.device)

    def sample_timesteps(self, size: int) -> torch.Tensor:
        '''
        randomly sample timesteps
        '''
        return torch.randint(0, self.num_inf_timesteps, (size, )).to(self.config.device)
    
    def prev_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        '''
        get prev timestep
        △t = train_steps / inference_steps
        return t - △t
        '''
        return timestep - self.num_train_timesteps // self.num_inf_timesteps


class DDPMScheduler(Scheduler):
    def __init__(self, config: 'Config') -> None:
        super().__init__(config)

    def step(self, 
             noise_pred: torch.Tensor,
             timestep: torch.Tensor,
             noisy_image: torch.Tensor) -> torch.Tensor:
        prev_t = self.prev_timestep(timestep)
        alpha_bar_at_t = self.alphas_cumprod[timestep]
        alpha_bar_at_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_bar_at_t = 1 - alpha_bar_at_t
        beta_bar_at_prev_t = 1 - alpha_bar_at_prev_t

        # α_bar_t / α_bar_(t-1)
        current_alpha_t = alpha_bar_at_t / alpha_bar_at_prev_t
        current_beta_t = 1 - current_alpha_t

        # predict x0
        denoised_image = (noisy_image - torch.sqrt(beta_bar_at_t) * noise_pred) / torch.sqrt(alpha_bar_at_t)
        # clip [-1, 1]
        denoised_image = denoised_image.clamp(-self.config.clip, self.config.clip)

        # predict μ
        pred_original_sample_coeff = (torch.sqrt(alpha_bar_at_prev_t) * current_beta_t) / beta_bar_at_t
        current_sample_coeff = torch.sqrt(current_alpha_t) * beta_bar_at_prev_t / beta_bar_at_t
        pred_prev_image = pred_original_sample_coeff * denoised_image + current_sample_coeff * noisy_image

        # add noise σ_t * z
        variance = 0
        if timestep > 0:
            z = torch.randn(noise_pred.shape).to(self.config.device)
            variance = (1 - alpha_bar_at_prev_t) / (1 - alpha_bar_at_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            variance = torch.sqrt(variance) * z

        return pred_prev_image + variance
    



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
        pred = model(noisy_image, timesteps)[0]
        loss = torch.nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        # clip grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress_bar.update(1)
        logs = {'loss': loss.detach().item(), 'epoch': epoch+1}
        progress_bar.set_postfix(**logs)

# save checkpoints
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, 'checkpoint/model_epoch' + str(config.epoch))
