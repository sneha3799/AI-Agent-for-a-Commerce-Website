import open_clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# trade-off is that CLIP's text understanding is shallower
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

# Embeddings 
def generate_embeddings(input, is_image=True):
    if is_image:
        image = Image.open(input).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
    else:
        text_input = tokenizer([input]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(text_input)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().tolist()