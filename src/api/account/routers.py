# from fastapi import APIRouter


# import os
# import sys
# import json
# from pathlib import Path
# from datetime import datetime, timedelta, timezone
# from enum import Enum
# from typing import Optional, List
# from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, FileResponse
# from fastapi.security import OAuth2PasswordBearer
# from jose import jwt, JWTError
# from passlib.context import CryptContext
# from pydantic import BaseModel, EmailStr, Field
# from loguru import logger
# import uvicorn

# # Optional libraries (install if you use them)
# import fitz  # PyMuPDF
# from docx import Document
# import pytesseract
# from PIL import Image

# # --- Project imports ---
# # Add repo root (adjusted for /src/* usage)
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from src.core.coder import MedicalCoder
# from src.core.em_codes import PatientType

# from src.api.em_api import (
#     f_path,TokenOut,
#     UserOut,
#     get_db,
#     get_current_user,
#     User,create_access_token,
#     verify_password,
#     LoginRequest,
#     SignupRequest,
#     app,
#     hash_password)
# # DB (SQLAlchemy 2.x)

# from sqlalchemy import select
# from sqlalchemy.orm import Session






# router = APIRouter()



# @router.get("/home", response_class=HTMLResponse)
# async def home_page():
#     html_path = os.path.join(f_path,"index.html")
#     # html_path = "web\index.html"
#     # if html_path.exists():
#     return FileResponse(html_path)
#     # return HTMLResponse(
#     #     "<h1>Medical Coding Automation API</h1><p>Frontend not found. API is running at /docs</p>"
#     # )

# @router.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "service": "Medical Coding Automation API",
#         "version": "1.0.0",
#     }


# # Auth Endpoints

# @router.post("/signup", response_model=UserOut)
# def signup(payload: SignupRequest, db: Session = Depends(get_db)):
#     if payload.confirm_password is not None and payload.password != payload.confirm_password:
#         raise HTTPException(status_code=400, detail="Passwords do not match")

#     existing = db.scalars(select(User).where(User.email == payload.email)).first()
#     if existing:
#         raise HTTPException(status_code=400, detail="Email already registered")

#     user = User(
#         email=payload.email,
#         name=payload.name.strip(),
#         password_hash=hash_password(payload.password),
#     )
#     db.add(user)
#     db.commit()
#     db.refresh(user)
#     return UserOut(id=user.id, email=user.email, name=user.name)

# @router.post("/login", response_model=TokenOut)
# def login(payload: LoginRequest, db: Session = Depends(get_db)):
#     user = db.scalars(select(User).where(User.email == payload.email)).first()
#     if not user or not verify_password(payload.password, user.password_hash):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = create_access_token(sub=str(user.id), email=user.email)
#     return TokenOut(access_token=token)


# # Me (JWT decode demo)

# @router.get("/me", response_model=UserOut)
# def me(user: User = Depends(get_current_user)):
#     return UserOut(id=user.id, email=user.email, name=user.name)


# def start_server(host: str = "0.0.0.0", port: int = 8000):
#     logger.info(f"Starting Medical Coding Automation API server on {host}:{port}")
#     uvicorn.run(app, host=host, port=port)

# if __name__ == "__main__":
#     start_server()