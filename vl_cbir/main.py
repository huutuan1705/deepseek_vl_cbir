import torch
import argparse
import torch.nn.functional as F 

from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import TripletMarginLoss
from uitls import get_dataloader
from model import VLM
from transformers import SiglipVisionModel, AutoModel

vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
text_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader):
    model.eval()
    test_cosine = []
    with torch.no_grad():
        for _, (pos_pixel_values, neg_pixel_values, input_ids, attention_mask) in enumerate(tqdm(test_loader)):
            pos_pixel_values = pos_pixel_values.to(device)
            neg_pixel_values = neg_pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image_proj, _, text_proj = model(pos_pixel_values, neg_pixel_values, input_ids, attention_mask)
            cosine_similarity_mean = F.cosine_similarity(image_proj, text_proj, dim=1)
            test_cosine.append(cosine_similarity_mean)
            
    cosine_mean = sum(test_cosine) / len(test_cosine)
    
    return cosine_mean

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Large vision language for caption based image retrieval')
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--t_max', type=int, default=10)
    parsers.add_argument('--root_data', type=str, default='./../')
    
    parsers.add_argument('--train_size', type=int, default=15000)
    parsers.add_argument('--test_size', type=int, default=4000)
    parsers.add_argument('--db_size', type=int, default=12783)
    parsers.add_argument('--output_dim', type=int, default=64)
    
    args = parsers.parse_args()
    train_loader, test_loader, search_loader = get_dataloader(args)
    model = VLM(args, vision_encoder, text_encoder)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.t_max)
    scaler = GradScaler()
    loss_fn = TripletMarginLoss(margin=args.margin)
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} / {args.epochs}")
        model.train()
        losses = []
        cosine_max = -100000
        for _, (pos_pixel_values, neg_pixel_values, input_ids, attention_mask) in enumerate(tqdm(train_loader, dynamic_ncols=True, ncols=100)):
            model.train()
            pos_pixel_values = pos_pixel_values.to(device)
            neg_pixel_values = neg_pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pos_image_proj, neg_image_proj, text_proj = model(pos_pixel_values, neg_pixel_values, input_ids, attention_mask)
            
            loss = loss_fn(pos_image_proj, text_proj, neg_image_proj)
            losses.append(loss.item())
            loss.backward()
            optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
        
        cosine_mean = evaluate(model, test_loader)   
        loss_mean = sum(losses) / len(losses)
        print('Training loss:       {:.4f}'.format(loss_mean))
        print('Test Cosine mean:    {:.4f}'.format(cosine_mean))
        
        if cosine_mean >= cosine_max:
            cosine_max = cosine_mean
            best_checkpoint = 'best_model.pth'
            torch.save(model, best_checkpoint)
            print(f"Best model at epoch {epoch+1}")
            
        last_checkpoint = 'last_model.pth'
        torch.save(model, last_checkpoint)