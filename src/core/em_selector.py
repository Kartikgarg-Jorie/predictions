"""     
E&M Code Selector

Selects appropriate E&M code based on:
1. Patient type (new vs established)
2. Medical Decision Making (MDM) complexity OR
3. Total time spent on the date of encounter

Per CPT guidelines, code can be selected by EITHER MDM OR time (whichever is more advantageous).
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from .em_codes import (
    PatientType,
    MDMLevel,
    EMCode,
    get_em_codes_by_patient_type
)
from .em_mdm_calculator import MDMElements, MDMCalculator


@dataclass
class EMEncounterData:
    """Data about an E&M encounter for code selection"""
    patient_type: PatientType
    mdm_elements: Optional[MDMElements] = None
    total_time_minutes: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate that at least MDM or time is provided"""
        if self.mdm_elements is None and self.total_time_minutes is None:
            return False
        return True


@dataclass
class EMCodeSelection:
    """Result of E&M code selection"""
    selected_code: EMCode
    selection_method: str  # "mdm" or "time"
    mdm_level: Optional[MDMLevel]
    time_minutes: Optional[int]
    confidence: str  # "high", "medium", "low"
    rationale: str


class EMCodeSelector:
    """
    Select appropriate E&M code for office/outpatient visits.
    
    Implements CPT 2021+ guidelines for code selection based on:
    - Medical Decision Making (MDM) complexity
    - Total time spent on date of encounter
    """
    
    def __init__(self):
        self.mdm_calculator = MDMCalculator()
    
    def select_code(self, encounter: EMEncounterData) -> EMCodeSelection:
        """
        Select appropriate E&M code based on encounter data.
        
        Args:
            encounter: EMEncounterData with patient type and either MDM elements or time
            
        Returns:
            EMCodeSelection with selected code and rationale
            
        Raises:
            ValueError: If encounter data is invalid
        """
        if not encounter.validate():
            raise ValueError("Encounter must include either MDM elements or total time")
        
        logger.info(f"Selecting E&M code for {encounter.patient_type.value} patient")
        
        # Get applicable codes for patient type
        applicable_codes = get_em_codes_by_patient_type(encounter.patient_type)
        
        # Try both methods if available and pick the higher-level code
        mdm_based_code = None
        time_based_code = None
        mdm_level = None
        
        if encounter.mdm_elements:
            mdm_level = self.mdm_calculator.calculate_mdm_level(encounter.mdm_elements)
            mdm_based_code = self._select_by_mdm(applicable_codes, mdm_level)
            logger.debug(f"MDM-based selection: {mdm_based_code.code if mdm_based_code else 'None'}")
        
        if encounter.total_time_minutes is not None:
            time_based_code = self._select_by_time(applicable_codes, encounter.total_time_minutes)
            logger.debug(f"Time-based selection: {time_based_code.code if time_based_code else 'None'}")
        
        # Select the more advantageous code (higher level)
        selected_code, method = self._choose_best_code(mdm_based_code, time_based_code)
        
        # Determine confidence level
        confidence = self._assess_confidence(encounter, selected_code, method)
        
        # Generate rationale
        rationale = self._generate_rationale(
            encounter,
            selected_code,
            method,
            mdm_level,
            mdm_based_code,
            time_based_code
        )
        
        logger.success(f"Selected code: {selected_code.code} via {method}")
        
        return EMCodeSelection(
            selected_code=selected_code,
            selection_method=method,
            mdm_level=mdm_level,
            time_minutes=encounter.total_time_minutes,
            confidence=confidence,
            rationale=rationale
        )
    
    def _select_by_mdm(self, codes: list[EMCode], mdm_level: MDMLevel) -> Optional[EMCode]:
        """Select code based on MDM level"""
        for code in codes:
            if code.mdm_level == mdm_level:
                return code
        return None
    
    def _select_by_time(self, codes: list[EMCode], time_minutes: int) -> Optional[EMCode]:
        """
        Select code based on total time.
        Time ranges are defined in CPT guidelines.
        """
        for code in codes:
            # Check if time falls within the code's time range
            if time_minutes >= code.time_minutes_min:
                if code.time_minutes_max is None or time_minutes <= code.time_minutes_max:
                    # For codes without max (highest level), return if time meets minimum
                    if code.time_minutes_max is None:
                        return code
                    # For codes with max, return if within range
                    elif time_minutes <= code.time_minutes_max:
                        return code
        
        # If time is below minimum, return lowest code
        return codes[0] if codes else None
    
    def _choose_best_code(
        self,
        mdm_code: Optional[EMCode],
        time_code: Optional[EMCode]
    ) -> tuple[EMCode, str]:
        """
        Choose the more advantageous code (per CPT guidelines).
        
        Returns:
            Tuple of (selected_code, selection_method)
        """
        if mdm_code is None and time_code is None:
            raise ValueError("No valid code could be selected")
        
        if mdm_code is None:
            return time_code, "time"
        
        if time_code is None:
            return mdm_code, "mdm"
        
        # Both available - compare code levels (higher code number = higher level)
        mdm_code_num = int(mdm_code.code)
        time_code_num = int(time_code.code)
        
        if time_code_num > mdm_code_num:
            return time_code, "time"
        elif mdm_code_num > time_code_num:
            return mdm_code, "mdm"
        else:
            # Same level - prefer MDM as primary method
            return mdm_code, "mdm"
    
    def _assess_confidence(
        self,
        encounter: EMEncounterData,
        selected_code: EMCode,
        method: str
    ) -> str:
        """Assess confidence level of code selection"""
        # High confidence if both MDM and time support the same or adjacent codes
        if encounter.mdm_elements and encounter.total_time_minutes:
            return "high"
        
        # Medium confidence if only one method available but clear selection
        if method == "mdm" and encounter.mdm_elements:
            return "medium"
        
        if method == "time" and encounter.total_time_minutes:
            # Check if time is close to boundaries
            if selected_code.time_minutes_max:
                mid_range = (selected_code.time_minutes_min + selected_code.time_minutes_max) / 2
                time_diff = abs(encounter.total_time_minutes - mid_range)
                if time_diff < 3:  # Within 3 minutes of mid-range
                    return "high"
            return "medium"
        
        return "low"
    
    def _generate_rationale(
        self,
        encounter: EMEncounterData,
        selected_code: EMCode,
        method: str,
        mdm_level: Optional[MDMLevel],
        mdm_code: Optional[EMCode],
        time_code: Optional[EMCode]
    ) -> str:
        """Generate human-readable rationale for code selection"""
        rationale_parts = []
        
        rationale_parts.append(f"Selected {selected_code.code} for {encounter.patient_type.value} patient")
        
        if method == "mdm" and mdm_level:
            rationale_parts.append(f"based on {mdm_level.value} complexity MDM")
        elif method == "time" and encounter.total_time_minutes:
            rationale_parts.append(f"based on {encounter.total_time_minutes} minutes total time")
        
        # Add comparison if both methods were available
        if mdm_code and time_code and mdm_code.code != time_code.code:
            other_method = "time" if method == "mdm" else "mdm"
            other_code = time_code if method == "mdm" else mdm_code
            rationale_parts.append(
                f"(higher than {other_code.code} suggested by {other_method} criteria)"
            )
        
        return ". ".join(rationale_parts) + "."


def select_em_code_simple(
    patient_type: PatientType,
    mdm_elements: Optional[MDMElements] = None,
    time_minutes: Optional[int] = None
) -> EMCodeSelection:
    """
    Convenience function for simple E&M code selection.
    
    Args:
        patient_type: NEW or ESTABLISHED patient
        mdm_elements: Optional MDM element assessment
        time_minutes: Optional total time in minutes
        
    Returns:
        EMCodeSelection with selected code
    """
    encounter = EMEncounterData(
        patient_type=patient_type,
        mdm_elements=mdm_elements,
        total_time_minutes=time_minutes
    )
    
    selector = EMCodeSelector()
    return selector.select_code(encounter)
