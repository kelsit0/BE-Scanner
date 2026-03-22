from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Analysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    patient_id: int = Field(foreign_key="user.id")
    doctor_id: int = Field(foreign_key="user.id")

    diagnosis: str
    report: str

    confirmed_diagnosis: Optional[str] = None
    is_confirmed: bool = Field(default=False)

    doctor_notes: Optional[str] = None

    image_name: str
    image_data: bytes

    created_at: datetime = Field(default_factory=datetime.utcnow)
