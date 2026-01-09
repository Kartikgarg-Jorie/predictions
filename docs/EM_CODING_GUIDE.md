# E&M Coding Implementation Guide

## Overview

This module implements **Evaluation and Management (E&M) coding** for office/outpatient visits based on **CPT 2021+ guidelines**. It supports automatic code selection using either:
1. **Medical Decision Making (MDM)** complexity, OR
2. **Total time** spent on the date of encounter

The system automatically selects the more advantageous code when both methods are available.

## 📋 Supported Codes

### New Patient Codes (99202-99205)
- **99202**: Straightforward MDM (15-29 minutes)
- **99203**: Low complexity MDM (30-44 minutes)
- **99204**: Moderate complexity MDM (45-59 minutes)
- **99205**: High complexity MDM (60+ minutes)

### Established Patient Codes (99211-99215)
- **99211**: Minimal MDM (0-9 minutes)
- **99212**: Straightforward MDM (10-19 minutes)
- **99213**: Low complexity MDM (20-29 minutes)
- **99214**: Moderate complexity MDM (30-39 minutes)
- **99215**: High complexity MDM (40+ minutes)

## 🎯 Medical Decision Making (MDM)

MDM complexity is determined by the **"2 out of 3" rule** - must meet criteria for 2 out of 3 elements:

### 1. Number and Complexity of Problems Addressed
- **Minimal**: 1 self-limited or minor problem
- **Low**: 2+ self-limited/minor problems OR 1 stable chronic illness
- **Moderate**: 1+ chronic illness with exacerbation OR 2+ stable chronic illnesses OR 1 undiagnosed new problem
- **High**: 1+ chronic illness with severe exacerbation OR acute/chronic illness posing threat to life

### 2. Amount/Complexity of Data Reviewed
- **Minimal/None**: No significant data review
- **Limited**: Review of external records OR independent historian
- **Moderate**: Independent interpretation of tests OR discussion with external physician
- **Extensive**: Both interpretation AND discussion OR extensive record review

### 3. Risk of Complications
- **Minimal**: No significant risk
- **Low**: Prescription drug management
- **Moderate**: Minor procedure or diagnostic procedure with risk OR decision for surgery
- **High**: Drug therapy requiring intensive monitoring OR parenteral controlled substances OR threat to life

## 💻 Usage Examples

### Example 1: Using the Main MedicalCoder Class

```python
from src.core.coder import MedicalCoder
from src.core.em_codes import PatientType
from src.core.em_mdm_calculator import (
    MDMElements,
    ProblemComplexity,
    DataComplexity,
    RiskLevel
)

# Initialize coder
coder = MedicalCoder()

# Define MDM elements
elements = MDMElements(
    problem_complexity=ProblemComplexity.MODERATE,
    data_complexity=DataComplexity.MODERATE,
    risk_level=RiskLevel.MODERATE
)

# Get code suggestion
result = coder.suggest_em_code(
    patient_type=PatientType.ESTABLISHED,
    mdm_elements=elements,
    time_minutes=35
)

print(f"Selected Code: {result.selected_code.code}")
print(f"Rationale: {result.rationale}")
# Output: Selected Code: 99214
```

### Example 2: Using Helper Functions

```python
from src.core.em_mdm_calculator import (
    determine_problem_complexity,
    determine_data_complexity,
    determine_risk_level
)

# Determine complexity from clinical factors
problem = determine_problem_complexity(
    num_problems=2,
    has_chronic_illness=True,
    chronic_illness_exacerbation=True
)  # Returns: ProblemComplexity.MODERATE

data = determine_data_complexity(
    reviewed_external_records=True,
    independent_interpretation=True
)  # Returns: DataComplexity.MODERATE

risk = determine_risk_level(
    prescription_drug_management=True,
    minor_procedure=True
)  # Returns: RiskLevel.MODERATE
```

### Example 3: Time-Based Selection

```python
# For prolonged visits (e.g., counseling), time can drive code selection
result = coder.suggest_em_code(
    patient_type=PatientType.ESTABLISHED,
    time_minutes=52  # Extended counseling session
)

# May select higher code based on time even with lower MDM
```

## 🧪 Testing

Run comprehensive tests:

```powershell
python -m pytest tests/test_em_coding.py -v
```

Test specific scenarios:

```powershell
python -m pytest tests/test_em_coding.py::TestClinicalScenarios -v
```

## 📊 Demo Application

Run the interactive demo with 6 realistic clinical scenarios:

```powershell
python examples/em_coding_demo.py
```

Scenarios include:
1. Simple URI visit (99211/99212)
2. Diabetes management (99213)
3. New patient with multiple conditions (99203)
4. COPD exacerbation (99215)
5. Acute chest pain (99205)
6. Prolonged counseling visit (time-based)

## 🏗️ Module Architecture

```
src/core/
├── em_codes.py              # CPT code definitions and enums
├── em_mdm_calculator.py     # MDM complexity calculation
├── em_selector.py           # Main code selection logic
└── coder.py                 # Integration into MedicalCoder class

tests/
└── test_em_coding.py        # Comprehensive test suite (31 tests)

examples/
└── em_coding_demo.py        # Interactive demonstration
```

## 🔧 API Reference

### Main Classes

#### `MedicalCoder`
Main interface for medical coding.

**Methods:**
- `suggest_em_code(patient_type, mdm_elements=None, time_minutes=None) -> EMCodeSelection`

#### `EMCodeSelector`
Core E&M code selection engine.

**Methods:**
- `select_code(encounter: EMEncounterData) -> EMCodeSelection`

#### `MDMCalculator`
Calculates MDM complexity level.

**Methods:**
- `calculate_mdm_level(elements: MDMElements) -> MDMLevel`
- `calculate_mdm_level_detailed(elements: MDMElements) -> dict`

### Data Classes

#### `EMCodeSelection`
Result of code selection containing:
- `selected_code: EMCode`
- `selection_method: str` (mdm/time)
- `mdm_level: Optional[MDMLevel]`
- `time_minutes: Optional[int]`
- `confidence: str` (high/medium/low)
- `rationale: str`

#### `MDMElements`
Container for MDM assessment:
- `problem_complexity: ProblemComplexity`
- `data_complexity: DataComplexity`
- `risk_level: RiskLevel`

## ✅ Compliance

This implementation follows:
- **CPT 2021+ Guidelines** for E&M office/outpatient visit codes
- **AMA CPT Manual** specifications
- **CMS E&M Documentation Guidelines**

### Key Guidelines Implemented:
✅ 2 out of 3 rule for MDM complexity  
✅ Time-based selection option  
✅ Selection of most advantageous code (MDM vs. time)  
✅ Separate code sets for new vs. established patients  
✅ Proper time ranges per CPT guidelines  

## 🚀 Future Enhancements

Planned features:
- [ ] NLP extraction of MDM elements from clinical notes
- [ ] Prolonged service codes (99417, 99418)
- [ ] Critical care codes (99291, 99292)
- [ ] Consultation codes
- [ ] Modifier recommendations
- [ ] Documentation template generation
- [ ] Compliance checking and audit support

## 📝 Clinical Scenarios Quick Reference

### Simple Visit (99211-99212)
- Single minor problem
- No extensive workup
- Minimal prescriptions

### Routine Follow-up (99213)
- Stable chronic condition
- Review labs/records
- Prescription management

### Complex Management (99214)
- Multiple chronic conditions
- Exacerbation
- Moderate risk treatment

### High Complexity (99215/99205)
- Life-threatening condition
- Extensive workup
- High-risk interventions

## 🤝 Contributing

When adding new E&M code types:
1. Define codes in `em_codes.py`
2. Implement selection logic in `em_selector.py`
3. Add tests in `tests/test_em_coding.py`
4. Update documentation

## 📚 References

- [CPT 2024 E&M Guidelines](https://www.ama-assn.org/practice-management/cpt/evaluation-and-management-em-office-or-other-outpatient-services)
- [CMS E&M Documentation](https://www.cms.gov/outreach-and-education/medicare-learning-network-mln/mlnproducts/downloads/eval-mgmt-serv-guide-icn006764.pdf)
- American Medical Association CPT Manual

---

**Last Updated**: October 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅
