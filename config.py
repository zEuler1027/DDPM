import torch

class Config:
    def __init__(self) -> None:
        self.proj_name = 'diff'
        
        # ddpm
        self.timesteps = 1000
        self.image_size = 64
        self.image_channels = 3
        self.beta0 = 1e-4 # start
        self.beta1 = 0.02 # end

        # train hyperparameters
        self.batch = 64
        self.lr = 1e-4
        self.epochs = 1
        self.save_period = 5
        self.sample_period =1
        
        # inference
        self.steps = 500
        self.num_inf_images = 32

        # model hyperparameters
        self.seed = 42
        self.base_channels = 64
        self.timestep_embed_dim = 64
        self.timestep_proj_dim = 256
        self.layers = 2
        self.input_channels = 1

        # others
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip = 1.0
