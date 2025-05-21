import math
import multiprocessing
import os.path
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from datasets import targetpad_transform, CIRDataset
from utils import collate_fn
from siglip_cir import Siglip_Retrieval

class CIRPlus(nn.Module):
    def __init__(self, blip_model_name, tau=0.01,
                 transform="targetpad", target_ratio=1.25, encoder='both',
                 device=torch.device('cuda'), plus=False):
        super().__init__()

        # initial main model
        self.device = device
        self.plus = plus
        self.siglip = Siglip_Retrieval()