from PIL import Image, ImageFile
import torch
from app.core.model import clip_preprocess, device

ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    tensor = clip_preprocess(image).unsqueeze(0).to(device)
    return tensor
