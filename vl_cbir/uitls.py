import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import SiglipProcessor, AutoTokenizer

from dataset import FlickrDataset

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

def get_transform(mode):
    if mode == 'train':
        transform_list = [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    return transforms.Compose(transform_list)

