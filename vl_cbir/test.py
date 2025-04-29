import torch
import argparse
from model import VLM
from uitls import load_image_from_url, find_caption, find_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Test LVL for caption based image retrieval')
    parsers.add_argument('--output_dim', type=int, default=64)
    parsers.add_argument('--pretrained', type=str, default="./../")
    parsers.add_argument('--url', type=str, default="https://cdn.shopify.com/s/files/1/0624/1746/9697/files/siberian-husky-100800827-2000-9449ca147e0e4b819bce5189c2411188_600x600.jpg?v=1690185264")
    parsers.add_argument('--caption', type=str, default="A cat is sitting down")

    args = parsers.parse_args()
    model = VLM(args)
    model.to(device)
    model.load_state_dict(args.pretrained)
    
    image = load_image_from_url(args.url)
    caption = args.caption
    
    image_vector = find_caption(image, model)
    caption_vector = find_image(caption, model)
    similarity = torch.cosine_similarity(image_vector, caption_vector)
    print(f"Cosine similarity giữa ảnh và caption: {similarity.item()}")