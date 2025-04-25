from transformers import SiglipVisionModel, SiglipProcessor, AutoModel, AutoTokenizer

siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def collate_fn(batch):
    images = [item['image'] for item in batch]
    captions = []
    for item in batch:
        if isinstance(item['caption'], str):
            captions.append(item['caption'])
        elif isinstance(item['caption'], list):
            captions.append(item['caption'][0])
    image_inputs = siglip_processor(images=images, return_tensors="pt", padding=True)
    text_inputs = deberta_tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    
    return image_inputs['pixel_values'], text_inputs['input_ids'], text_inputs['attention_mask']
