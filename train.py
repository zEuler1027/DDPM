from config import Config
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as v2
from torchvision.datasets import MNIST


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
            return image.to(self.config.device), label.to(self.config.device)
        else:
            return image.to(self.config.device)
        
train_dataset = MINISTData(config, 'data', return_label=True)
# print(train_dataset[0][0].shape) [1, 64, 64]

