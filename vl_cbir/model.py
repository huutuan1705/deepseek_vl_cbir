import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SiglipVisionModel, AutoModel

class MLPProjection(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super(MLPProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = F.normalize(x)
        
        return x
    
class VLM(nn.Module):
    def __init__(self, args):
        super(VLM, self).__init__()
        self.args = args
        self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.text_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        self.img_linear = MLPProjection(output_dim=self.args.output_dim)
        self.text_linear = MLPProjection(output_dim=self.args.output_dim)
        
    def forward(self, pixel_values_pos, pixel_values_neg, input_ids, attention_mask):
        vision_outputs_pos = self.vision_encoder(pixel_values_pos)
        image_pos_features = vision_outputs_pos.pooler_output
        
        vision_outputs_neg = self.vision_encoder(pixel_values_neg)
        image_neg_features = vision_outputs_neg.pooler_output
        
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        image_pos_proj = self.img_linear(image_pos_features)
        image_neg_proj = self.img_linear(image_neg_features)
        text_proj = self.text_linear(text_features)
        
        return image_pos_proj, image_neg_proj, text_proj
    