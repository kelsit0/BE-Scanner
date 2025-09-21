from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, ImageFile
import torch
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
        # Abrir imagen recibida
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error cargando la imagen: {e}")

    # Preprocesar para CLIP
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy()

    # Predecir clase
    pred_label = clf.predict(embedding)[0]
    clase = idx_to_class[pred_label]

    # Generar explicación con BioGPT
    prompt = f"This is a chest X-ray classified as {clase.lower()}. Describe what this condition means in a clinical context:"
    inputs = biogpt_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = biogpt_model.generate(**inputs, max_new_tokens=100)
    explicacion = biogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "diagnostico": clase,
        "explicacion_medica": explicacion
    }

#router = APIRouter()

#@router.post("/analyze")
#async def analyze_image(file: UploadFile = File(...)):
#    contents = await file.read()  # bytes de la imagen
#    size_kb = round(len(contents) / 1024, 2)
#    return {
#        "filename": file.filename,
#        "content_type": file.content_type,
#        "size_kb": size_kb,
#        "message": "Imagen recibida correctamente 🚀"
#    }
