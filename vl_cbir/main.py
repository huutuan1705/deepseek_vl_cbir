import torch
import argparse

from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from uitls import get_dataloader
from model import VLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, args):
    model.eval()
    return

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Large vision language for caption based image retrieval')
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--t_max', type=int, default=10)
    parsers.add_argument('--root_data', type=str, default='./../')
    
    args = parsers.parse_args()
    train_loader, test_loader, search_loader = get_dataloader(args)
    model = VLM(args)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.t_max)
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
    