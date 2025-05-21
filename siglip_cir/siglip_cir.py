import torch
import torch.nn.functional as F

from torch import nn
from transformers import SiglipVisionModel,  AutoModel, AutoTokenizer
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2ForMaskedLM

visual_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
text_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
config.is_decoder = True
config.add_cross_attention = True

def init_tokenizer():
    deberta_tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    deberta_tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    deberta_tokenizer.enc_token_id = deberta_tokenizer.additional_special_tokens_ids[0]
    return deberta_tokenizer

class Siglip_Retrieval(nn.Module):
    def __init__(self, args):
        super(Siglip_Retrieval, self).__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = DebertaV2Model(config)
        
        self.vision_proj = nn.Linear(args.img_feature_size, args.embed_dim)  # 768 -> 256
        self.text_proj = nn.Linear(args.text_feature_size, args.embed_dim)  # 768 -> 256
        
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
    def img_embed(self, image, atts=False, return_pool_and_normalized=False):
        '''
        If in train: return pooled_and_normalized features;
        if in val: return raw features (for computing txt-img fusion)
            and pooled_and_normalized_features (for all target images)
        '''
        image_embeds = self.visual_encoder(image)  
        out = (image_embeds,)
        if return_pool_and_normalized:
            image_embeds_p = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # B x 256
            out += (image_embeds_p,)
        if atts:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            out += (image_atts,)
        if len(out) == 1:
            out = out[0]  # if only one type of feature is returned, unwrap the tuple
        return out
    
    def img_txt_fusion(self, r_image_embeds, t_image_embeds, text, train=False, return_raw=False):
        device = r_image_embeds.device

        r_image_atts = torch.ones(r_image_embeds.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(device)
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        output_pos = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=r_image_embeds,
                                       encoder_attention_mask=r_image_atts,
                                       return_dict=True,
                                       )
        
        # compute logits
        predicted_features = F.normalize(self.text_proj(output_pos.last_hidden_state[:, 0, :]), dim=-1)  # B x 256
        if not train:
            if return_raw:
                return output_pos
            else:
                return predicted_features
        else:
            target_features = t_image_embeds  # already normalized
            logits = predicted_features @ target_features.T / self.temp  # B x B
            return logits