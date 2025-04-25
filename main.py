import argparse
from dataset import FlickrDataset

def get_dataloader(args):
    dataset_train = FlickrDataset(args, mode="train")

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Large vision language for caption based image retrieval')
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--root_data', type=str, default='./../')
    
    args = parsers.parse_args()