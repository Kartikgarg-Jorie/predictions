# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Essential Commands

### Setup
```powershell
python -m venv venv                          # Create virtual environment
venv\Scripts\activate                        # Activate virtual environment
pip install -r requirements.txt              # Install dependencies
cp config.example.py config.py               # Create config file (edit as needed)
```

### Running the Application
```powershell
python main.py                               # Run main application with sample data
python src/api/server.py                     # Start FastAPI server (when implemented)
```

### Testing
```powershell
pytest tests/                                # Run all tests
pytest tests/test_coder.py                   # Run specific test file
pytest tests/ -v                             # Run with verbose output
pytest tests/ --cov=src --cov-report=html    # Run with coverage report
pytest tests/test_coder.py::TestMedicalCoder::test_initialization  # Run specific test
```

### Code Quality
```powershell
black src/ tests/                            # Format code
isort src/ tests/                            # Sort imports
flake8 src/ tests/                           # Lint code
mypy src/                                    # Type checking
```

## Architecture Overview

### Purpose
Healthcare automation tool that uses NLP and ML to suggest medical codes (ICD-10, CPT, HCPCS) from clinical documentation, improving coding accuracy and reducing manual effort.

### Core Components

**1. Medical Coder Engine** (`src/core/coder.py`)
- `MedicalCoder` class: Main interface for code suggestion
- `suggest_codes()`: Processes clinical text and returns suggested codes
- `validate_codes()`: Validates codes for compliance and accuracy
- Currently in early development - core methods not yet implemented

**2. NLP Pipeline** (`src/nlp/` - planned)
- Extract medical entities from unstructured clinical notes
- Uses spaCy, transformers, and MedCAT for medical text processing
- Identify diagnoses, procedures, symptoms from documentation

**3. Validation Engine** (`src/validation/` - planned)
- Cross-reference codes for CMS compliance
- Check payer-specific requirements
- Validate code combinations and hierarchies

**4. API Layer** (`src/api/` - planned)
- FastAPI-based REST API for EHR integration
- Async processing support with uvicorn
- Pydantic models for request/response validation

### Technology Stack
- **NLP/ML**: spaCy, transformers (Hugging Face), PyTorch, scikit-learn, MedCAT
- **API**: FastAPI, Uvicorn, Pydantic
- **Database**: SQLAlchemy with Alembic migrations
- **Testing**: pytest with async support
- **Logging**: Loguru for enhanced logging
- **Code Quality**: black, isort, flake8, mypy

## Development Workflow

### Project Structure
```
src/
├── core/           # Business logic for medical coding
├── models/         # Data models (planned)
├── nlp/            # NLP processing modules (planned)
├── validation/     # Code validation logic (planned)
└── api/            # API endpoints (planned)

tests/              # Unit and integration tests
data/               # Sample clinical data and datasets
models/             # Trained ML models
docs/               # Documentation
```

### Current State
This is an early-stage project with:
- Basic project structure established
- Core `MedicalCoder` class skeleton created
- Test infrastructure set up
- Dependencies defined for NLP/ML pipeline

**Main implementation tasks remaining**:
- NLP pipeline for clinical text extraction
- ICD-10/CPT/HCPCS code suggestion logic
- Validation and compliance checking
- FastAPI REST API development
- ML model training pipeline

### Adding New Features

**When implementing code suggestion**:
1. Add NLP extraction logic to identify medical concepts
2. Map concepts to appropriate code systems (ICD-10, CPT, HCPCS)
3. Return structured results with confidence scores
4. Update `suggest_codes()` in `src/core/coder.py`

**When implementing validation**:
1. Load code databases for current year versions (configurable in `config.py`)
2. Implement rule-based validation for code combinations
3. Check against CMS guidelines
4. Update `validate_codes()` in `src/core/coder.py`

### Configuration
- Copy `config.example.py` to `config.py` before running
- Key settings: NLP models, code versions, API configuration, validation thresholds
- Use environment variables for secrets (e.g., `SECRET_KEY`)

### Medical Code Systems
- **ICD-10**: International Classification of Diseases (diagnoses)
- **CPT**: Current Procedural Terminology (procedures)
- **HCPCS**: Healthcare Common Procedure Coding System (supplies, services)

Version configuration in `config.py`:
- `ICD10_VERSION = "2024"`
- `CPT_VERSION = "2024"`
- `HCPCS_VERSION = "2024"`

## Testing Strategy

### Test Organization
- `tests/test_coder.py`: Tests for core MedicalCoder class
- Additional test files should mirror `src/` structure (e.g., `tests/test_nlp.py` for `src/nlp/`)
- Use pytest fixtures for reusable test data (clinical notes, expected codes)

### Testing Guidelines
- Test with realistic clinical documentation samples
- Validate code suggestions against known correct codes
- Test edge cases: incomplete notes, ambiguous terminology, multi-condition cases
- Mock external dependencies (ML models, databases) for unit tests

## Important Notes

### Medical Compliance
This tool assists medical coders but should not replace professional judgment. All suggested codes require review by qualified medical coding professionals before submission to payers.

### Data Privacy
Clinical documentation contains PHI (Protected Health Information). Ensure HIPAA compliance:
- Do not commit real patient data to version control
- Use synthetic/anonymized data for testing
- Implement appropriate access controls in production

### Dependencies
The project uses specialized medical NLP libraries:
- **MedCAT**: Medical Concept Annotation Tool for clinical text
- **spaCy**: General NLP with medical model support
- **transformers**: BERT-based models for clinical context understanding
