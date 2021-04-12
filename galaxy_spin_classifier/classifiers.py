import os
import torch
from .resnet import resnet18

__all__ = ['ZSClassifier']


cd = os.path.dirname(__file__)
ZS_MODEL = os.path.join(cd, 'data/zs-4-4-resnet18-beta-5/epoch_200.p')


class ZSClassifier:
    """
    Finding Z-like (clockwise) and S-like (anticlockwise) spiral galaxies.
    """
    def __init__(self, state_dict=ZS_MODEL, device='cuda'):
        self.model = resnet18(num_channels=1, num_classes=2, use_max_pool=False, use_avg_pool=True, add_fc=[1024, 512, 256])
        if device == 'cuda':
            device = torch.device(device)
            self.model.load_state_dict(torch.load(state_dict))
            self.model.to(device)
        elif device == 'cpu':
            device = torch.device(device)
            self.model.load_state_dict(torch.load(state_dict, map_location=device))
