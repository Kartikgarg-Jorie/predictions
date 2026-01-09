
"""
Medical Coding Automation API
- Auth (signup/login, JWT)
- E&M coding endpoints
- Upload + NLP extraction with persistence
"""

from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import create_engine, select, String, Integer, ForeignKey, Text
from src.nlp.clinical_extractor import load_text_from_file
import tempfile
from main import (
    RevokedToken,
    DATABASE_URL,
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    get_current_user,
    oauth2_scheme,
    decode_token,
    get_db
)

from src.nlp.clinical_extractor import ClinicalNLPExtractor
from src.core.em_mdm_calculator import (
    MDMElements,
    ProblemComplexity,
    DataComplexity,
    RiskLevel,
)
from src.core.em_codes import PatientType
from src.core.coder import MedicalCoder
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from loguru import logger
import uvicorn

# Optional libraries (install if you use them)
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# App

app = FastAPI(
    title="Medical Coding Automation API",
    description="E&M Coding API for healthcare providers",
    version="1.0.0",
)

# CORS (allow all; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mount
# if static_dir.exists():
f_path = os.path.join(os.path.dirname(__file__), "../../web")
app.mount("/static", StaticFiles(directory=f_path), name="static")


# DB init

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith(
        "sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255), default="")
    notes: Mapped[List["ClinicalNote"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
class ClinicalNote(Base):
    __tablename__ = "clinical_notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    content_text: Mapped[str] = mapped_column(
        Text)  # raw text decoded from upload
    extractor_result_json: Mapped[str] = mapped_column(
        Text)  # JSON string for portability
    confidence_score: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    user: Mapped[User] = relationship(back_populates="notes")


Base.metadata.create_all(bind=engine)

# Pydantic Schemas (Auth)
class SignupRequest(BaseModel):
    name: str = Field(min_length=2)
    email: EmailStr
    password: str = Field(min_length=6)
    # keep optional to support old clients
    confirm_password: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str

class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class PatientTypeEnum(str, Enum):
    NEW = "new"
    ESTABLISHED = "established"

class ProblemComplexityEnum(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class DataComplexityEnum(str, Enum):
    MINIMAL_NONE = "minimal_or_none"
    LIMITED = "limited"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"

class RiskLevelEnum(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class MDMRequest(BaseModel):
    patient_type: PatientTypeEnum
    problem_complexity: ProblemComplexityEnum
    data_complexity: DataComplexityEnum
    risk_level: RiskLevelEnum
    time_minutes: Optional[int] = Field(None, ge=0, le=300)

class TimeBasedRequest(BaseModel):
    patient_type: PatientTypeEnum
    time_minutes: int = Field(..., ge=1, le=300)


class QuickAssessmentRequest(BaseModel):
    patient_type: PatientTypeEnum
    num_problems: int = Field(1, ge=1, le=10)
    has_chronic_illness: bool = False
    chronic_illness_exacerbation: bool = False
    chronic_illness_severe: bool = False
    life_threatening_condition: bool = False
    reviewed_external_records: bool = False
    independent_interpretation: bool = False
    discussed_with_external_physician: bool = False
    prescription_drug_management: bool = False
    decision_for_surgery: bool = False
    drug_therapy_requiring_monitoring: bool = False
    acute_threat_to_life: bool = False
    time_minutes: Optional[int] = Field(None, ge=0, le=300)

class CodeSuggestionResponse(BaseModel):
    code: str
    description: str
    patient_type: str
    selection_method: str
    mdm_level: Optional[str]
    time_minutes: Optional[int]
    confidence: str
    rationale: str

# Services (Coder + NLP)
coder = MedicalCoder()
extractor = ClinicalNLPExtractor(learning_enabled=True)

@app.get("/", response_class=HTMLResponse)
async def login_page():
    login_html_path = os.path.join(f_path, "login.html")
    return FileResponse(login_html_path)

@app.get("/", response_class=HTMLResponse)
async def login_page():
    login_html_path = os.path.join(f_path, "login.html")
    return FileResponse(login_html_path)
    
@app.get("/home", response_class=HTMLResponse)
async def home_page():
    html_path = os.path.join(f_path, "index.html")
    return FileResponse(html_path)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Medical Coding Automation API",
        "version": "1.0.0",
    }
# Auth Endpoints

@app.post("/signup", response_model=UserOut)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    if payload.confirm_password is not None and payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    existing = db.scalars(select(User).where(
        User.email == payload.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email,
        name=payload.name.strip(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, email=user.email, name=user.name)


@app.post("/login", response_model=TokenOut)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalars(select(User).where(User.email == payload.email)).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Useremail or Password may incorrect")
    token = create_access_token(sub=str(user.id), email=user.email)
    refresh_token = create_refresh_token(sub=str(user.id), email=user.email)
    return TokenOut(access_token=token, refresh_token=refresh_token)

# Me (JWT decode demo)
@app.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email, name=user.name)


@app.post("/api/em/suggest", response_model=CodeSuggestionResponse)
async def suggest_em_code_mdm(request: MDMRequest):
    try:
        patient_type = PatientType.NEW if request.patient_type == PatientTypeEnum.NEW else PatientType.ESTABLISHED

        mdm_elements = MDMElements(
            problem_complexity=ProblemComplexity[request.problem_complexity.upper(
            )],
            data_complexity=DataComplexity[
                request.data_complexity.upper().replace("MINIMAL_OR_NONE", "MINIMAL_NONE")
            ],
            risk_level=RiskLevel[request.risk_level.upper()],
        )

        result = coder.suggest_em_code(
            patient_type=patient_type,
            mdm_elements=mdm_elements,
            time_minutes=request.time_minutes,
        )

        return CodeSuggestionResponse(
            code=result.selected_code.code,
            description=result.selected_code.description,
            patient_type=patient_type.value,
            selection_method=result.selection_method,
            mdm_level=result.mdm_level.value if result.mdm_level else None,
            time_minutes=result.time_minutes,
            confidence=result.confidence,
            rationale=result.rationale,
        )

    except Exception as e:
        logger.error(f"Error in suggest_em_code_mdm: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/em/suggest-time", response_model=CodeSuggestionResponse)
async def suggest_em_code_time(request: TimeBasedRequest):
    try:
        patient_type = PatientType.NEW if request.patient_type == PatientTypeEnum.NEW else PatientType.ESTABLISHED
        result = coder.suggest_em_code(
            patient_type=patient_type, time_minutes=request.time_minutes)

        return CodeSuggestionResponse(
            code=result.selected_code.code,
            description=result.selected_code.description,
            patient_type=patient_type.value,
            selection_method=result.selection_method,
            mdm_level=result.mdm_level.value if result.mdm_level else None,
            time_minutes=result.time_minutes,
            confidence=result.confidence,
            rationale=result.rationale,
        )

    except Exception as e:
        logger.error(f"Error in suggest_em_code_time: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/em/suggest-quick", response_model=CodeSuggestionResponse)
async def suggest_em_code_quick(request: QuickAssessmentRequest):
    try:
        from src.core.em_mdm_calculator import (
            determine_problem_complexity,
            determine_data_complexity,
            determine_risk_level,
        )

        patient_type = PatientType.NEW if request.patient_type == PatientTypeEnum.NEW else PatientType.ESTABLISHED

        problem = determine_problem_complexity(
            num_problems=request.num_problems,
            has_chronic_illness=request.has_chronic_illness,
            chronic_illness_exacerbation=request.chronic_illness_exacerbation,
            chronic_illness_severe=request.chronic_illness_severe,
            life_threatening_condition=request.life_threatening_condition,
        )

        data = determine_data_complexity(
            reviewed_external_records=request.reviewed_external_records,
            independent_interpretation=request.independent_interpretation,
            discussed_with_external_physician=request.discussed_with_external_physician,
        )

        risk = determine_risk_level(
            prescription_drug_management=request.prescription_drug_management,
            decision_for_surgery=request.decision_for_surgery,
            drug_therapy_requiring_monitoring=request.drug_therapy_requiring_monitoring,
            acute_threat_to_life=request.acute_threat_to_life,
        )

        mdm_elements = MDMElements(
            problem_complexity=problem, data_complexity=data, risk_level=risk)

        result = coder.suggest_em_code(
            patient_type=patient_type,
            mdm_elements=mdm_elements,
            time_minutes=request.time_minutes,
        )

        return CodeSuggestionResponse(
            code=result.selected_code.code,
            description=result.selected_code.description,
            patient_type=patient_type.value,
            selection_method=result.selection_method,
            mdm_level=result.mdm_level.value if result.mdm_level else None,
            time_minutes=result.time_minutes,
            confidence=result.confidence,
            rationale=result.rationale,
        )

    except Exception as e:
        logger.error(f"Error in suggest_em_code_quick: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Upload + NLP extraction (persist to DB)

@app.post("/api/em/upload-note")
async def upload_clinical_note(

    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    try:
        content_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name

        text = load_text_from_file(tmp_path)

        # Run NLP extraction
        result = extractor.extract_from_note(text)
        confidence = float(result.get("confidence_score", 0.0))
        # print(confidence)

        # Persist in DB
        note = ClinicalNote(
            user_id=user.id,
            filename=file.filename,
            content_text=text,
            extractor_result_json=json.dumps(result, ensure_ascii=False),
            confidence_score=confidence,
        )
        db.add(note)
        db.commit()
        db.refresh(note)

        return {
            "status": "success",
            "message": "Clinical note analyzed and saved.",
            "note_id": note.id,
            "user_id": user.id,
            "filename": file.filename,
            "confidence_score": confidence,
            "extracted_elements": {
                "patient_type": result.get("patient_type"),
                "num_problems": result.get("num_problems"),
                "has_chronic_illness": result.get("has_chronic_illness"),
                "chronic_illness_exacerbation": result.get("chronic_illness_exacerbation"),
                "chronic_illness_severe": result.get("chronic_illness_severe"),
                "life_threatening_condition": result.get("life_threatening_condition"),
                "reviewed_external_records": result.get("reviewed_external_records"),
                "independent_interpretation": result.get("independent_interpretation"),
                "discussed_with_external_physician": result.get("discussed_with_external_physician"),
                "prescription_drug_management": result.get("prescription_drug_management"),
                "decision_for_surgery": result.get("decision_for_surgery"),
                "drug_therapy_requiring_monitoring": result.get("drug_therapy_requiring_monitoring"),
                "acute_threat_to_life": result.get("acute_threat_to_life"),
                "icd_codes": result.get("icd_codes"),
                "cpt_codes": result.get("cpt_codes"),
                "medications": result.get("medications"),
            },
        }

    except Exception as e:
        logger.error(f"Error in upload_clinical_note: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing note: {str(e)}")


# Codes list

@app.get("/api/codes/list")
async def list_em_codes():
    from src.core.em_codes import ALL_EM_OFFICE_CODES

    codes = [
        {
            "code": code.code,
            "description": code.description,
            "patient_type": code.patient_type.value,
            "mdm_level": code.mdm_level.value if code.mdm_level else None,
            "typical_time": code.typical_time_minutes,
        }
        for code in ALL_EM_OFFICE_CODES
    ]
    return {"codes": codes}
# Notes viewing (for current user)

@app.get("/api/em/my-notes")
def list_my_notes(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    notes = (
        db.query(ClinicalNote)
        .filter(ClinicalNote.user_id == user.id)
        .order_by(ClinicalNote.created_at.desc())
        .all()
    )
    return {
        "count": len(notes),
        "notes": [
            {
                "id": n.id,
                "filename": n.filename,
                "confidence_score": n.confidence_score,
                "created_at": n.created_at.isoformat(),
            }
            for n in notes
        ],
    }


@app.get("/api/em/my-notes/{note_id}")
def get_my_note(note_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = db.query(ClinicalNote).filter(
        ClinicalNote.id == note_id, ClinicalNote.user_id == user.id
    ).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return {
        "id": note.id,
        "filename": note.filename,
        "confidence_score": note.confidence_score,
        "created_at": note.created_at.isoformat(),
        "content_text": note.content_text,
        "extractor_result": json.loads(note.extractor_result_json),
    }

# logout

@app.post("/logout")
def logout(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db),
):
    try:
        payload = decode_token(token)
        jti = payload.get("jti")
        if not jti:
            raise HTTPException(status_code=400, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    existing = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
    if not existing:
        db.add(RevokedToken(jti=jti))
        db.commit()

    return {"message": "Logged out successfully"}

# Server start (CLI)
def start_server(host: str = "0.0.0.0", port: int = 8000):
    logger.info(
        f"Starting Medical Coding Automation API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()