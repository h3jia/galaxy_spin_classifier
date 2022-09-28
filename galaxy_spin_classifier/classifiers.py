import os
import torch
from .resnet import *

__all__ = ['ZSClassifier']


cd = os.path.dirname(__file__)
ZS_DICT = os.path.join(cd, 'data/zs-27-00-120.p')

# (hejia@nia) hejia@nia-login06:~/galaxy_spin_classifier$ cp ~/galaxy-spin-data/galaxy-zoo-2/torch/10/00-resnet34-alpha-1/models/epoch-400.p galaxy_spin_classifier/data/zs-10-00-400.p
# (hejia@nia) hejia@nia-login06:~/galaxy_spin_classifier$ cp ~/galaxy-spin-data/galaxy-zoo-2/torch/15/00-resnet34/models/epoch-200.p galaxy_spin_classifier/data/zs-15-00-200.p
# (hejia@nia) hejia@nia-login07:~/galaxy_spin_classifier$ cp ~/galaxy-spin-data/galaxy-zoo-2/torch/27/00/models/epoch-120.p galaxy_spin_classifier/data/zs-27-00-120.p


class ZSClassifier:
    """
    Finding Z-like (clockwise) and S-like (anticlockwise) spiral galaxies.
    """
    def __init__(self, model='resnet50', state_dict=ZS_DICT, device='cuda'):
        if not model == 'resnet50':
            raise NotImplementedError
        if isinstance(model, str):
            model = eval(model)
        self.model = model(num_channels=3, num_classes=2, use_max_pool=True, use_avg_pool=True,
                           avg_pool_size=(1, 1), add_fc=[512, 512, 64, 64])
        if device == 'cuda':
            device = torch.device(device)
            self.model.load_state_dict(torch.load(state_dict))
            self.model.to(device)
        elif device == 'cpu':
            device = torch.device(device)
            self.model.load_state_dict(torch.load(state_dict, map_location=device))
        self.eval()

    def __call__(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)
