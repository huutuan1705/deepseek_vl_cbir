
from torch.utils.data import Dataset

class FlickrDataset(Dataset):
    def __init__(self, args, mode):
        super(FlickrDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.db_size = args.db_size
        
    def __len__(self):
        if self.mode == "train":
            return self.train_size
        if self.mode == "test":
            return self.test_size
        return self.db_size