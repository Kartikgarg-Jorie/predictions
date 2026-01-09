
"""
E&M (Evaluation and Management) Code Definitions

Based on CPT 2021+ guidelines for office/outpatient E&M services.
Codes are selected based on either:
1. Medical Decision Making (MDM) complexity, OR
2. Total time spent on the date of encounter

Reference: CPT 2024 E&M Guidelines
"""
# for MDM Entry
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class PatientType(Enum):
    """Patient type for E&M coding"""
    NEW = "new"
    ESTABLISHED = "established"

class MDMLevel(Enum):
    """Medical Decision Making complexity levels"""
    MINIMAL = "minimal"  
    STRAIGHTFORWARD = "straightforward"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class ProblemComplexity(Enum):
    """Number and complexity of problems addressed"""
    MINIMAL = "minimal"  # 1 self-limited or minor problem
    LOW = "low"  # 2+ self-limited/minor problems OR 1 stable chronic illness
    MODERATE = "moderate"  # 1+ chronic illness with exacerbation OR 2+ stable chronic illnesses OR 1 undiagnosed new problem with uncertain prognosis
    HIGH = "high"  # 1+ chronic illness with severe exacerbation OR 1 acute or chronic illness posing threat to life/bodily function

class DataComplexity(Enum):
    """Amount and complexity of data to be reviewed and analyzed"""
    MINIMAL_NONE = "minimal_or_none"
    LIMITED = "limited"  # Review of tests/documents from external source OR independent historian
    MODERATE = "moderate"  # Independent interpretation of tests OR discussion with external physician
    EXTENSIVE = "extensive"  # Independent interpretation + discussion OR extensive review of records

class RiskLevel(Enum):
    """Risk of complications and/or morbidity or mortality of patient management"""
    MINIMAL = "minimal"
    LOW = "low"  # Low risk of morbidity from additional diagnostic testing or treatment
    MODERATE = "moderate"  # Moderate risk from diagnostic procedures or treatment
    HIGH = "high"  # High risk of morbidity from additional diagnostic testing or treatment

@dataclass
class EMCode:
    """E&M Code definition with criteria"""
    code: str
    description: str
    patient_type: PatientType
    mdm_level: Optional[MDMLevel]
    time_minutes_min: int
    time_minutes_max: Optional[int]
    typical_time_minutes: int
    
    def __str__(self):
        return f"{self.code}: {self.description} (MDM: {self.mdm_level.value if self.mdm_level else 'N/A'}, Time: {self.typical_time_minutes} min)"

EM_NEW_PATIENT_CODES = [
    EMCode(
        code="99202",
        description="Office/outpatient visit, new patient, straightforward MDM",
        patient_type=PatientType.NEW,
        mdm_level=MDMLevel.STRAIGHTFORWARD,
        time_minutes_min=15,
        time_minutes_max=29,
        typical_time_minutes=20
    ),

    EMCode(
        code="99203",
        description="Office/outpatient visit, new patient, low complexity MDM",
        patient_type=PatientType.NEW,
        mdm_level=MDMLevel.LOW,
        time_minutes_min=30,
        time_minutes_max=44,
        typical_time_minutes=35
    ),

    EMCode(
        code="99204",
        description="Office/outpatient visit, new patient, moderate complexity MDM",
        patient_type=PatientType.NEW,
        mdm_level=MDMLevel.MODERATE,
        time_minutes_min=45,
        time_minutes_max=59,
        typical_time_minutes=50
    ),
    EMCode(
        code="99205",
        description="Office/outpatient visit, new patient, high complexity MDM",
        patient_type=PatientType.NEW,
        mdm_level=MDMLevel.HIGH,
        time_minutes_min=60,
        time_minutes_max=None,
        typical_time_minutes=75
    ),
]
# Office/Outpatient E&M Codes - ESTABLISHED PATIENTS (99211-99215)
EM_ESTABLISHED_PATIENT_CODES = [
    EMCode(
        code="99211",
        description="Office/outpatient visit, established patient, minimal MDM",
        patient_type=PatientType.ESTABLISHED,
        mdm_level=MDMLevel.MINIMAL,
        time_minutes_min=0,
        time_minutes_max=9,
        typical_time_minutes=5
    ),
    EMCode(
        code="99212",
        description="Office/outpatient visit, established patient, straightforward MDM",
        patient_type=PatientType.ESTABLISHED,
        mdm_level=MDMLevel.STRAIGHTFORWARD,
        time_minutes_min=10,
        time_minutes_max=19,
        typical_time_minutes=15
    ),
    EMCode(
        code="99213",
        description="Office/outpatient visit, established patient, low complexity MDM",
        patient_type=PatientType.ESTABLISHED,
        mdm_level=MDMLevel.LOW,
        time_minutes_min=20,
        time_minutes_max=29,
        typical_time_minutes=25
    ),
    
    EMCode(
        code="99214",
        description="Office/outpatient visit, established patient, moderate complexity MDM",
        patient_type=PatientType.ESTABLISHED,
        mdm_level=MDMLevel.MODERATE,
        time_minutes_min=30,
        time_minutes_max=39,
        typical_time_minutes=35
    ),
    EMCode(
        code="99215",
        description="Office/outpatient visit, established patient, high complexity MDM",
        patient_type=PatientType.ESTABLISHED,
        mdm_level=MDMLevel.HIGH,
        time_minutes_min=40,
        time_minutes_max=None,
        typical_time_minutes=50
    ),
]

# Combined lookup
ALL_EM_OFFICE_CODES = EM_NEW_PATIENT_CODES + EM_ESTABLISHED_PATIENT_CODES

def get_em_codes_by_patient_type(patient_type: PatientType) -> list[EMCode]:
    """Get E&M codes for a specific patient type"""
    if patient_type == PatientType.NEW:
        return EM_NEW_PATIENT_CODES
    else:
        return EM_ESTABLISHED_PATIENT_CODES
