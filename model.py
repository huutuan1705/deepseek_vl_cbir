import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SiglipVisionModel, SiglipProcessor, AutoModel, AutoTokenizer

class MLPProjection(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super(MLPProjection).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = F.normalize(x)
        
        return x
    
class VLM(nn.Module):
    def __init__(self, ):
        super().__init__()