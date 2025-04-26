import torch
import argparse

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import FlickrDataset
from uitls import get_dataloader
from model import VLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Large vision language for caption based image retrieval')
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--root_data', type=str, default='./../')
    
    args = parsers.parse_args()
    train_loader, test_loader, search_loader = get_dataloader(args)
    model = VLM(args)
    model.to(device)
    
    