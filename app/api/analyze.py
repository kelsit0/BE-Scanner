from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()  # bytes de la imagen
    size_kb = round(len(contents) / 1024, 2)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_kb": size_kb,
        "message": "Imagen recibida correctamente 🚀"
    }
