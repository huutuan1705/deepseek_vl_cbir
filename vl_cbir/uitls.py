import torch
import torch.nn as nn
import requests

from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SiglipProcessor, AutoTokenizer

from dataset import FlickrDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def collate_fn(batch):
    pos_imgs = [item['pos_img'] for item in batch]
    neg_imgs = [item['neg_img'] for item in batch]
    captions = []
    for item in batch:
        if isinstance(item['caption'], str):
            captions.append(item['caption'])
        elif isinstance(item['caption'], list):
            captions.append(item['caption'][0])
    pos_image_inputs = siglip_processor(images=pos_imgs, return_tensors="pt", padding=True)
    neg_image_inputs = siglip_processor(images=neg_imgs, return_tensors="pt", padding=True)
    text_inputs = deberta_tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    
    return pos_image_inputs['pixel_values'], neg_image_inputs['pixel_values'], \
                text_inputs['input_ids'], text_inputs['attention_mask']

def get_dataloader(args):
    dataset_train = FlickrDataset(args, mode="train")
    dataset_test = FlickrDataset(args, mode="test")
    dataset_search = FlickrDataset(args, mode="search")
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.threads, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.threads, collate_fn=collate_fn)
    search_loader = DataLoader(dataset_search, batch_size=args.batch_size, num_workers=args.threads, collate_fn=collate_fn)
    
    return train_loader, test_loader, search_loader

def find_caption(image, model):
    model.eval()
    with torch.no_grad():
        image_inputs = siglip_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(device)
        image_features = model.vision_encoder(pixel_values).pooler_output
        image_proj = model.img_linear(image_features)
        image_proj = nn.functional.normalize(image_proj, p=2, dim=-1)
        return image_proj

def find_image(caption, model):
    model.eval()
    with torch.no_grad():
        text_inputs = deberta_tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)
        text_features = model.text_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        text_proj = model.text_linear(text_features)
        text_proj = nn.functional.normalize(text_proj, p=2, dim=-1)
        return text_proj
    
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")