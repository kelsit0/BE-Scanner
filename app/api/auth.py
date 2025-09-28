from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.models.user import User
from app.core.config import get_session
from app.core.security import hash_password, verify_password

router = APIRouter()

@router.post("/register")
def register(username: str, password: str, session: Session = Depends(get_session)):
    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already registered")

    new_user = User(username=username, password=hash_password(password))
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return {"message": "User registered", "user": new_user.username}

@router.post("/login")
def login(username: str, password: str, session: Session = Depends(get_session)):
    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    return {"message": f"Welcome {user.username} 🚀"}
