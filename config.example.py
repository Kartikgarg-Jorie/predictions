"""
Configuration file template for Medical Coding Automation Tool

Copy this file to config.py and update with your settings.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Application settings
APP_NAME = "Medical Coding Automation Tool"
APP_VERSION = "0.1.0"
DEBUG = True

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Database settings
DATABASE_URL = "sqlite:///./medical_coding.db"
# For PostgreSQL: "postgresql://user:password@localhost/dbname"

# NLP Model settings
NLP_MODEL = "en_core_web_sm"  # spaCy model
TRANSFORMER_MODEL = "bert-base-uncased"  # HuggingFace model

# Medical coding settings
ICD10_VERSION = "2024"
CPT_VERSION = "2024"
HCPCS_VERSION = "2024"

# Validation settings
STRICT_VALIDATION = False
MIN_CONFIDENCE_THRESHOLD = 0.75

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

# Feature flags
ENABLE_ML_SUGGESTIONS = True
ENABLE_RULE_BASED_VALIDATION = True
ENABLE_AUDIT_LOGGING = True

# External API keys (if needed)
# EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# Cache settings
ENABLE_CACHE = True
CACHE_TTL = 3600  # seconds
