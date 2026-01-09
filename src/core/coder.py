"""
Medical Coder - Core class for automated medical code suggestion
"""

from typing import Dict, List, Optional, Any
from loguru import logger

from .em_codes import PatientType
from .em_mdm_calculator import MDMElements
from .em_selector import EMCodeSelector, EMEncounterData, EMCodeSelection


class MedicalCoder:
    """
    Main class for medical coding automation.
    
    Processes clinical documentation and suggests appropriate medical codes
    including E&M, ICD-10, CPT, and HCPCS codes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Medical Coder
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.em_selector = EMCodeSelector()
        logger.info("Medical Coder initialized with E&M coding support")
    
    def suggest_em_code(
        self,
        patient_type: PatientType,
        mdm_elements: Optional[MDMElements] = None,
        time_minutes: Optional[int] = None
    ) -> EMCodeSelection:
        """
        Suggest E&M (Evaluation and Management) code for office/outpatient visit.
        
        Args:
            patient_type: NEW or ESTABLISHED patient
            mdm_elements: Optional MDM (Medical Decision Making) assessment
            time_minutes: Optional total time spent in minutes
            
        Returns:
            EMCodeSelection with selected code and rationale
            
        """
        logger.info(f"Suggesting E&M code for {patient_type.value} patient")
        
        encounter = EMEncounterData(
            patient_type=patient_type,
            mdm_elements=mdm_elements,
            total_time_minutes=time_minutes
        )
        
        return self.em_selector.select_code(encounter)
    
    def suggest_codes(self, clinical_text: str) -> Dict[str, List[str]]:
        """
        Suggest medical codes based on clinical documentation.
        
        NOTE: This method is for future NLP-based code extraction.
        For now, use suggest_em_code() directly for E&M codes.
        
        Args:
            clinical_text: Clinical note or documentation text
            
        Returns:
            Dictionary with suggested codes by type (E&M, ICD-10, CPT, HCPCS)
        """
        logger.debug(f"Processing text of length: {len(clinical_text)}")
      
        
        raise NotImplementedError(
            "Automated code extraction from clinical text not yet implemented. "
            "Use suggest_em_code() method directly for E&M coding."
        )
    
    def validate_codes(self, codes: List[str]) -> Dict[str, bool]:
        """
        Validate medical codes for compliance and accuracy
        
        Args:
            codes: List of medical codes to validate
            
        Returns:
            Dictionary mapping codes to validation results
        """
        logger.debug(f"Validating {len(codes)} codes")
        
        raise NotImplementedError("Code validation not yet implemented")
