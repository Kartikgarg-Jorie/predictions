"""
Medical Decision Making (MDM) Calculator for E&M Coding

Per CPT 2021+ guidelines, MDM level is determined by meeting 2 out of 3 elements:
1. Number and complexity of problems addressed
2. Amount and/or complexity of data to be reviewed and analyzed
3. Risk of complications and/or morbidity or mortality of patient management

MDM Levels: Straightforward, Low, Moderate, High
"""

from typing import Optional
from dataclasses import dataclass
from loguru import logger

from .em_codes import (
    MDMLevel,
    ProblemComplexity,
    DataComplexity,
    RiskLevel
)


@dataclass
class MDMElements:
    """Container for the three MDM elements"""
    problem_complexity: ProblemComplexity
    data_complexity: DataComplexity
    risk_level: RiskLevel


class MDMCalculator:
    """
    Calculate Medical Decision Making complexity level for E&M coding.
    
    Based on CPT "2 out of 3" rule:
    - Must meet or exceed criteria for 2 out of 3 elements to qualify for that MDM level
    """
    
    # MDM Element Mappings
    # Each element can map to different MDM levels
    
    PROBLEM_TO_MDM = {
        ProblemComplexity.MINIMAL: MDMLevel.STRAIGHTFORWARD,
        ProblemComplexity.LOW: MDMLevel.LOW,
        ProblemComplexity.MODERATE: MDMLevel.MODERATE,
        ProblemComplexity.HIGH: MDMLevel.HIGH,
    }
    
    DATA_TO_MDM = {
        DataComplexity.MINIMAL_NONE: MDMLevel.STRAIGHTFORWARD,
        DataComplexity.LIMITED: MDMLevel.LOW,
        DataComplexity.MODERATE: MDMLevel.MODERATE,
        DataComplexity.EXTENSIVE: MDMLevel.HIGH,
    }
    
    RISK_TO_MDM = {
        RiskLevel.MINIMAL: MDMLevel.STRAIGHTFORWARD,
        RiskLevel.LOW: MDMLevel.LOW,
        RiskLevel.MODERATE: MDMLevel.MODERATE,
        RiskLevel.HIGH: MDMLevel.HIGH,
    }
    
    # MDM Level hierarchy for comparison
    MDM_HIERARCHY = {
        MDMLevel.MINIMAL: 0,
        MDMLevel.STRAIGHTFORWARD: 1,
        MDMLevel.LOW: 2,
        MDMLevel.MODERATE: 3,
        MDMLevel.HIGH: 4,
    }
    
    def calculate_mdm_level(self, elements: MDMElements) -> MDMLevel:
        """
        Calculate MDM level using the "2 out of 3" rule.
        
        Args:
            elements: MDMElements containing problem, data, and risk assessments
            
        Returns:
            MDMLevel based on meeting 2 out of 3 criteria
        """
        # Map each element to its MDM level
        problem_mdm = self.PROBLEM_TO_MDM[elements.problem_complexity]
        data_mdm = self.DATA_TO_MDM[elements.data_complexity]
        risk_mdm = self.RISK_TO_MDM[elements.risk_level]
        
        logger.debug(f"MDM Elements - Problem: {problem_mdm.value}, Data: {data_mdm.value}, Risk: {risk_mdm.value}")
        
        # Convert to hierarchy values for comparison
        element_levels = [
            (problem_mdm, self.MDM_HIERARCHY[problem_mdm]),
            (data_mdm, self.MDM_HIERARCHY[data_mdm]),
            (risk_mdm, self.MDM_HIERARCHY[risk_mdm])
        ]
        
        # Sort by hierarchy value (ascending)
        element_levels.sort(key=lambda x: x[1])
        
        # The MDM level is determined by the MIDDLE value (2 out of 3 rule)
        # If two or more elements meet a level, that level qualifies
        final_mdm = element_levels[1][0]  # Middle element
        
        logger.info(f"Calculated MDM Level: {final_mdm.value}")
        return final_mdm
    
    def calculate_mdm_level_detailed(self, elements: MDMElements) -> dict:
        """
        Calculate MDM level with detailed breakdown.
        
        Returns:
            Dictionary with MDM level and detailed element analysis
        """
        problem_mdm = self.PROBLEM_TO_MDM[elements.problem_complexity]
        data_mdm = self.DATA_TO_MDM[elements.data_complexity]
        risk_mdm = self.RISK_TO_MDM[elements.risk_level]
        
        final_mdm = self.calculate_mdm_level(elements)
        
        return {
            "mdm_level": final_mdm,
            "elements": {
                "problem_complexity": {
                    "input": elements.problem_complexity.value,
                    "mdm_contribution": problem_mdm.value
                },
                "data_complexity": {
                    "input": elements.data_complexity.value,
                    "mdm_contribution": data_mdm.value
                },
                "risk_level": {
                    "input": elements.risk_level.value,
                    "mdm_contribution": risk_mdm.value
                }
            },
            "rule_applied": "2_out_of_3"
        }


def determine_problem_complexity(
    num_problems: int,
    has_chronic_illness: bool = False,
    chronic_illness_exacerbation: bool = False,
    chronic_illness_severe: bool = False,
    life_threatening_condition: bool = False,
    undiagnosed_uncertain_prognosis: bool = False
) -> ProblemComplexity:
    """
    Helper function to determine problem complexity based on clinical factors.
    
    Args:
        num_problems: Number of problems addressed
        has_chronic_illness: Patient has chronic illness
        chronic_illness_exacerbation: Chronic illness with exacerbation
        chronic_illness_severe: Chronic illness with severe exacerbation
        life_threatening_condition: Condition poses threat to life/bodily function
        undiagnosed_uncertain_prognosis: Undiagnosed new problem with uncertain prognosis
        
    Returns:
        ProblemComplexity level
    """
    # HIGH: Severe exacerbation or life-threatening
    if chronic_illness_severe or life_threatening_condition:
        return ProblemComplexity.HIGH
    
    # MODERATE: Exacerbation, 2+ stable chronic, or undiagnosed uncertain
    if chronic_illness_exacerbation or (has_chronic_illness and num_problems >= 2) or undiagnosed_uncertain_prognosis:
        return ProblemComplexity.MODERATE
    
    # LOW: 2+ minor problems or 1 stable chronic
    if num_problems >= 2 or has_chronic_illness:
        return ProblemComplexity.LOW
    
    # MINIMAL: 1 self-limited or minor problem
    return ProblemComplexity.MINIMAL


def determine_data_complexity(
    reviewed_external_records: bool = False,
    independent_interpretation: bool = False,
    discussed_with_external_physician: bool = False,
    independent_historian: bool = False,
    extensive_record_review: bool = False
) -> DataComplexity:
    """
    Helper function to determine data complexity based on documentation review.
    
    Args:
        reviewed_external_records: Reviewed tests/documents from external source
        independent_interpretation: Independent interpretation of tests/images
        discussed_with_external_physician: Discussion with external physician
        independent_historian: Obtained history from independent historian
        extensive_record_review: Extensive review of records from multiple sources
        
    Returns:
        DataComplexity level
    """
    # EXTENSIVE: Both interpretation AND discussion, OR extensive review
    if (independent_interpretation and discussed_with_external_physician) or extensive_record_review:
        return DataComplexity.EXTENSIVE
    
    # MODERATE: Independent interpretation OR discussion
    if independent_interpretation or discussed_with_external_physician:
        return DataComplexity.MODERATE
    
    # LIMITED: External records reviewed OR independent historian
    if reviewed_external_records or independent_historian:
        return DataComplexity.LIMITED
    
    # MINIMAL/NONE: No significant data review
    return DataComplexity.MINIMAL_NONE


def determine_risk_level(
    prescription_drug_management: bool = False,
    minor_procedure: bool = False,
    diagnostic_procedure_with_risk: bool = False,
    drug_therapy_requiring_monitoring: bool = False,
    decision_for_surgery: bool = False,
    parenteral_controlled_substances: bool = False,
    acute_threat_to_life: bool = False
) -> RiskLevel:
    """
    Helper function to determine risk level based on treatment/management.
    
    Args:
        prescription_drug_management: Prescription drug management
        minor_procedure: Decision for minor procedure with identified patient/procedure risk
        diagnostic_procedure_with_risk: Diagnostic procedure with identified risk
        drug_therapy_requiring_monitoring: Drug therapy requiring intensive monitoring
        decision_for_surgery: Decision regarding elective major surgery
        parenteral_controlled_substances: Parenteral controlled substances
        acute_threat_to_life: Acute or chronic illness posing threat to life
        
    Returns:
        RiskLevel
    """
    # HIGH: Threat to life or intensive monitoring/parenteral controlled substances
    if acute_threat_to_life or drug_therapy_requiring_monitoring or parenteral_controlled_substances:
        return RiskLevel.HIGH
    
    # MODERATE: Surgery decision or procedures with risk
    if decision_for_surgery or diagnostic_procedure_with_risk or minor_procedure:
        return RiskLevel.MODERATE
    
    # LOW: Prescription drug management
    if prescription_drug_management:
        return RiskLevel.LOW
    
    # MINIMAL: No significant risk
    return RiskLevel.MINIMAL
