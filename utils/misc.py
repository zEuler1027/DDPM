import torch
import numpy as np
import random
from datetime import datetime


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def date() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

if __name__ == '__main__':
    print(type(date()))
