import os
os.environ["HF_HOME"] = "D:/huggingface"
import torch
import open_clip
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP
clip_model_name = "ViT-B-32"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    clip_model_name, pretrained="openai"
)
clip_model = clip_model.to(device)

# BioGPT
biogpt_name = "microsoft/BioGPT-Large-PubMedQA"
biogpt_tokenizer = AutoTokenizer.from_pretrained(biogpt_name)
biogpt_model = AutoModelForCausalLM.from_pretrained(biogpt_name).to(device)
