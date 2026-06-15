import open_clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# trade-off is that CLIP's text understanding is shallower
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
# model.to(device)

# ✅ Load lazily on first use
_model = None
_preprocess = None
_tokenizer = None

def _load_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        _tokenizer = open_clip.get_tokenizer('ViT-B-32')
        _model.to(device)
    return _model, _preprocess, _tokenizer

# Embeddings
def generate_embeddings(input, is_image=True):
    model, preprocess, tokenizer = _load_model()
    
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