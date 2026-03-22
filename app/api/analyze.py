from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, ImageFile
import torch
import re
from app.core.classifier import clip_model, clip_preprocess, device, clf, idx_to_class
from transformers import AutoTokenizer, AutoModelForCausalLM
from sqlmodel import Session, select
from app.models.user import User
from app.core.config import get_session
from fastapi import Depends
from app.models.analysis import Analysis
from fastapi.responses import Response
import io
from sqlalchemy import func

from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os
import io


# Permitir imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

# Cargar BioGPT
biogpt_name = "microsoft/BioGPT-Large-PubMedQA"
biogpt_tokenizer = AutoTokenizer.from_pretrained(biogpt_name)
biogpt_model = AutoModelForCausalLM.from_pretrained(biogpt_name).to(device)

@router.post("/analyze")
async def analyze_image( username: str, patient_username: str, file: UploadFile = File(...), session: Session = Depends(get_session)):
    import io

    statement = select(User).where(User.username == username)
    doctor = session.exec(statement).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can analyze images")

    patient_statement = select(User).where(User.username == patient_username)
    patient = session.exec(patient_statement).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if patient.role != "patient":
        raise HTTPException(status_code=400, detail="Selected user is not a patient")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {e}")

    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy()

    pred_label = clf.predict(embedding)[0]
    clase = idx_to_class[pred_label]

    if clase.upper() == "PNEUMONIA":
        explicacion = (
            "Increased pulmonary opacity in the lower lung zone. "
            "Findings consistent with focal consolidation."
        )

    elif clase.upper() == "NORMAL":
        explicacion = (
            "Clear lung fields without focal consolidation. "
            "Cardiac silhouette and mediastinal contours within normal limits."
        )

    else:
        explicacion = (
            "No acute cardiopulmonary abnormalities identified. "
            "Radiographic appearance within expected limits."
        )

    analysis = Analysis(
        patient_id=patient.id,
        doctor_id=doctor.id,
        diagnosis=clase,
        report=explicacion,
        image_name=file.filename,
        image_data=image_bytes
    )

    session.add(analysis)
    session.commit()
    session.refresh(analysis)

    return {
        "analysis_id": analysis.id,
        "diagnostico_ia": clase,
        "reporte_generado": explicacion
    }

@router.get("/patient-history")
def patient_history(username: str,session: Session = Depends(get_session)):
    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access their history")

    from app.models.analysis import Analysis

    analyses = session.exec(
        select(Analysis).where(Analysis.patient_id == user.id)
    ).all()

    return [
    {
        "id": a.id,
        "diagnosis": a.diagnosis,
        "report": a.report,
        "doctor_notes": a.doctor_notes,
        "image_name": a.image_name,
        "created_at": a.created_at,
        "doctor_id": a.doctor_id,
        "patient_id": a.patient_id
    }
    for a in analyses]



@router.get("/doctor/patient-history")
def doctor_patient_history(doctor_username: str,patient_username: str,session: Session = Depends(get_session)):
    
    # Validar doctor
    doctor_statement = select(User).where(User.username == doctor_username)
    doctor = session.exec(doctor_statement).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient histories")

    # Buscar paciente
    patient_statement = select(User).where(User.username == patient_username)
    patient = session.exec(patient_statement).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if patient.role != "patient":
        raise HTTPException(status_code=400, detail="Selected user is not a patient")

    # Obtener historial
    from app.models.analysis import Analysis

    analyses = session.exec(select(Analysis).where(Analysis.patient_id == patient.id).order_by(Analysis.created_at.desc())).all()

    return [
    {
        "id": a.id,
        "diagnosis": a.diagnosis,
        "report": a.report,
        "doctor_notes": a.doctor_notes,
        "image_name": a.image_name,
        "created_at": a.created_at,
        "doctor_id": a.doctor_id,
        "patient_id": a.patient_id
    }
    for a in analyses]


@router.get("/analysis-image/{analysis_id}")
def get_analysis_image(analysis_id: int,username: str,session: Session = Depends(get_session)):

    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    from app.models.analysis import Analysis

    analysis = session.get(Analysis, analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Control acceso
    if user.role == "patient" and analysis.patient_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return Response(
        content=analysis.image_data,
        media_type="image/png"
    )
@router.put("/analysis/{analysis_id}/doctor-notes")
def add_doctor_notes(analysis_id: int,doctor_username: str,notes: str,session: Session = Depends(get_session)):
    # Validar doctor
    doctor_statement = select(User).where(User.username == doctor_username)
    doctor = session.exec(doctor_statement).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can add notes")

    # Buscar análisis
    analysis = session.get(Analysis, analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Verificar que el análisis pertenece al doctor
    if analysis.doctor_id != doctor.id:
        raise HTTPException(status_code=403, detail="You are not assigned to this analysis")

    # Guardar notas
    analysis.doctor_notes = notes

    session.add(analysis)
    session.commit()
    session.refresh(analysis)

    return {"message": "Doctor notes updated"}

@router.get("/doctor/patients")
def doctor_patients( doctor_username: str, session: Session = Depends(get_session)):
    # Validar doctor
    doctor_statement = select(User).where(User.username == doctor_username)
    doctor = session.exec(doctor_statement).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access this resource")

    # Obtener pacientes únicos atendidos por este doctor
    analyses = session.exec(
        select(Analysis).where(Analysis.doctor_id == doctor.id)
    ).all()

    # Extraer pacientes únicos
    patient_ids = set(a.patient_id for a in analyses)

    patients = session.exec(
        select(User).where(User.id.in_(patient_ids))
    ).all()

    return [
        {
            "id": p.id,
            "username": p.username
        }
        for p in patients
    ]
@router.put("/analysis/{analysis_id}/confirm-diagnosis")
def confirm_diagnosis(analysis_id: int,doctor_username: str, confirmed_diagnosis: str, session: Session = Depends(get_session)):
    # Validar doctor
    doctor_statement = select(User).where(User.username == doctor_username)
    doctor = session.exec(doctor_statement).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can confirm diagnosis")

    # Buscar análisis
    analysis = session.get(Analysis, analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Verificar que pertenece al doctor
    if analysis.doctor_id != doctor.id:
        raise HTTPException(status_code=403, detail="Not authorized for this analysis")

    # Confirmar
    analysis.confirmed_diagnosis = confirmed_diagnosis
    analysis.is_confirmed = True

    session.add(analysis)
    session.commit()
    session.refresh(analysis)

    return {"message": "Diagnosis confirmed"}

@router.get("/doctor/dashboard")
def doctor_dashboard( doctor_username: str, session: Session = Depends(get_session)):
    # Validar doctor
    doctor_statement = select(User).where(User.username == doctor_username)
    doctor = session.exec(doctor_statement).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access dashboard")

    # Total análisis
    total_analyses = session.exec(
        select(func.count()).select_from(Analysis)
        .where(Analysis.doctor_id == doctor.id)
    ).one()

    # Pacientes únicos
    patient_ids = session.exec(
        select(Analysis.patient_id)
        .where(Analysis.doctor_id == doctor.id)
    ).all()

    unique_patients = len(set(patient_ids))

    # Diagnósticos más frecuentes (IA original)
    diagnoses = session.exec(
        select(Analysis.diagnosis)
        .where(Analysis.doctor_id == doctor.id)
    ).all()

    diagnosis_count = {}
    for d in diagnoses:
        diagnosis_count[d] = diagnosis_count.get(d, 0) + 1

    # Confirmados
    confirmed_count = session.exec(
        select(func.count())
        .select_from(Analysis)
        .where(Analysis.doctor_id == doctor.id)
        .where(Analysis.is_confirmed == True)
    ).one()

    return {
        "total_analyses": total_analyses,
        "unique_patients": unique_patients,
        "diagnosis_distribution": diagnosis_count,
        "confirmed_diagnoses": confirmed_count
    }

@router.get("/analysis/{analysis_id}/export-pdf")
def export_pdf( analysis_id: int, username: str, session: Session = Depends(get_session)):
    # Validar usuario
    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    analysis = session.get(Analysis, analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Control acceso
    if user.role == "patient" and analysis.patient_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Crear archivo temporal
    file_path = f"report_{analysis_id}.pdf"

    doc = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()

    final_diagnosis = (
        analysis.confirmed_diagnosis
        if analysis.is_confirmed
        else analysis.diagnosis
    )

    elements.append(Paragraph("Radiological Analysis Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Date:</b> {analysis.created_at}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"<b>AI Diagnosis:</b> {analysis.diagnosis}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"<b>Final Diagnosis:</b> {final_diagnosis}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Imaging Findings:</b>", styles["Normal"]))
    elements.append(Spacer(1, 0.1 * inch))

    for line in analysis.report.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))

    if analysis.doctor_notes:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("<b>Doctor Notes:</b>", styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(analysis.doctor_notes, styles["Normal"]))

    doc.build(elements)

    return FileResponse(file_path, media_type="application/pdf", filename=file_path)
@router.get("/doctor/all-patients")
def get_all_patients(session: Session = Depends(get_session)):

    from app.models.user import User

    patients = session.exec(
        select(User).where(User.role == "patient")
    ).all()

    return [
        {
            "id": p.id,
            "username": p.username
        }
        for p in patients
    ]