from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, ImageFile
import torch
import re
from app.core.classifier import clip_model, clip_preprocess, device, clf, idx_to_class
from transformers import AutoTokenizer, AutoModelForCausalLM

# Permitir imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

# Cargar BioGPT
biogpt_name = "microsoft/BioGPT-Large-PubMedQA"
biogpt_tokenizer = AutoTokenizer.from_pretrained(biogpt_name)
biogpt_model = AutoModelForCausalLM.from_pretrained(biogpt_name).to(device)

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {e}")

    # Preprocesar con CLIP
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy()

    pred_label = clf.predict(embedding)[0]
    clase = idx_to_class[pred_label]

    # Prompt más restrictivo
    prompt = (
        f"You are a radiologist. The following chest X-ray was classified as: {clase}. "
        f"Provide ONLY objective imaging findings. "
        f"Write exactly 2-3 bullet points, each short (max 12 words). "
        f"Do NOT mention patients, doctors, treatments, CT or history."# or diagnosis. "
        f"mention the diagnosis in 12 words ONLY"
        f"Examples:\n"
        f"- No pulmonary infiltrates or consolidations.\n"
        f"- Normal cardiac silhouette and diaphragm.\n"
        f"- Opacity in the right lobe consistent with consolidation.\n\n"
        f"Answer:\n-"
    )

    inputs = biogpt_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = biogpt_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=2.0,
        eos_token_id=biogpt_tokenizer.eos_token_id
    )

    explicacion = biogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in explicacion:
        explicacion = explicacion.split("Answer:")[-1].strip()

    # Limpiar
    explicacion = re.sub(r"<.*?>", "", explicacion).strip()

    # Convertir en viñetas
    bullets = re.split(r"\n-+", explicacion)
    bullets = [b.strip(" -") for b in bullets if b.strip()]

    # Filtro: solo frases cortas y sin palabras prohibidas
    prohibidas = ["patient", "treatment", "history", "doctor", "CT", "ultrasound"]
    bullets = [
        b for b in bullets
        if len(b.split()) <= 12 and not any(p.lower() in b.lower() for p in prohibidas)
    ]

    # Tomar máximo 3
    explicacion = "\n- " + "\n- ".join(bullets[:3]) if bullets else "- No clear findings described."

    return {
        "diagnostico": clase,
        "explicacion_medica": explicacion
    }
