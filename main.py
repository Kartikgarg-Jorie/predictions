
"""
Medical Coding Automation Tool
Main entry point for the application
"""

import sys
from pathlib import Path
import os
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, create_engine
from passlib.context import CryptContext
from jose import jwt
from loguru import logger
from pydantic import BaseModel, EmailStr, Field
from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI, Depends, HTTPException
from jose import jwt, JWTError
from sqlalchemy.orm import (
    sessionmaker,
    Session,
    declarative_base
)



app = FastAPI(title="Medical Coding Automation Tool")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

# Security (bcrypt + JWT)

sys.path.insert(0, str(Path(__file__).parent))
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-prod")  # set in env for prod
JWT_ALG = "HS256"
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")  # used by Swagger /docs

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(*, sub: str, email: str, expires_minutes: int = JWT_EXP_MINUTES) -> str:
    payload = {
        "sub": sub,
        "email": email,
        "jti": str(uuid.uuid4()),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def create_refresh_token(*, sub: str, email: str,expires_minutes: int = JWT_EXP_MINUTES) -> str:
    payload = {
        "sub": sub,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXP_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.get(User, int(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Pydantic Schemas
class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    confirm_password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

#  code for logout and revoke class 
class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), unique=True, index=True, nullable=False)
    revoked_at = Column(
        String,
        default=lambda: datetime.now(timezone.utc).isoformat()
    )

def ensure_token_not_revoked(jti: str, db):
    revoked = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
    if revoked:
        raise HTTPException(status_code=401, detail="Token revoked")

def main():
    """Main application entry point"""
    logger.info("Starting Medical Coding Automation Tool")
    logger.info("Application completed")

if __name__ == "__main__":
    main()