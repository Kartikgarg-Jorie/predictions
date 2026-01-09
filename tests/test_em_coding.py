"""
Tests for E&M (Evaluation and Management) coding modules
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.em_codes import (
    PatientType,
    MDMLevel,
    ProblemComplexity,
    DataComplexity,
    RiskLevel,
    get_em_codes_by_patient_type
)
from src.core.em_mdm_calculator import (
    MDMElements,
    MDMCalculator,
    determine_problem_complexity,
    determine_data_complexity,
    determine_risk_level
)
from src.core.em_selector import (
    EMEncounterData,
    EMCodeSelector,
    select_em_code_simple
)


class TestEMCodes:
    """Test E&M code definitions"""
    
    def test_new_patient_codes_count(self):
        """Test that we have 4 new patient codes (99202-99205)"""
        codes = get_em_codes_by_patient_type(PatientType.NEW)
        assert len(codes) == 4
        assert codes[0].code == "99202"
        assert codes[3].code == "99205"
    
    def test_established_patient_codes_count(self):
        """Test that we have 5 established patient codes (99211-99215)"""
        codes = get_em_codes_by_patient_type(PatientType.ESTABLISHED)
        assert len(codes) == 5
        assert codes[0].code == "99211"
        assert codes[4].code == "99215"
    
    def test_code_mdm_levels(self):
        """Test that codes have correct MDM levels"""
        new_codes = get_em_codes_by_patient_type(PatientType.NEW)
        assert new_codes[0].mdm_level == MDMLevel.STRAIGHTFORWARD  # 99202
        assert new_codes[1].mdm_level == MDMLevel.LOW  # 99203
        assert new_codes[2].mdm_level == MDMLevel.MODERATE  # 99204
        assert new_codes[3].mdm_level == MDMLevel.HIGH  # 99205


class TestMDMCalculator:
    """Test Medical Decision Making calculator"""
    
    def test_straightforward_mdm(self):
        """Test straightforward MDM calculation"""
        calculator = MDMCalculator()
        elements = MDMElements(
            problem_complexity=ProblemComplexity.MINIMAL,
            data_complexity=DataComplexity.MINIMAL_NONE,
            risk_level=RiskLevel.MINIMAL
        )
        mdm = calculator.calculate_mdm_level(elements)
        assert mdm == MDMLevel.STRAIGHTFORWARD
    
    def test_low_mdm(self):
        """Test low complexity MDM (2 out of 3 at low level)"""
        calculator = MDMCalculator()
        elements = MDMElements(
            problem_complexity=ProblemComplexity.LOW,
            data_complexity=DataComplexity.LIMITED,
            risk_level=RiskLevel.MINIMAL
        )
        mdm = calculator.calculate_mdm_level(elements)
        assert mdm == MDMLevel.LOW
    
    def test_moderate_mdm(self):
        """Test moderate complexity MDM"""
        calculator = MDMCalculator()
        elements = MDMElements(
            problem_complexity=ProblemComplexity.MODERATE,
            data_complexity=DataComplexity.MODERATE,
            risk_level=RiskLevel.LOW
        )
        mdm = calculator.calculate_mdm_level(elements)
        assert mdm == MDMLevel.MODERATE
    
    def test_high_mdm(self):
        """Test high complexity MDM"""
        calculator = MDMCalculator()
        elements = MDMElements(
            problem_complexity=ProblemComplexity.HIGH,
            data_complexity=DataComplexity.EXTENSIVE,
            risk_level=RiskLevel.HIGH
        )
        mdm = calculator.calculate_mdm_level(elements)
        assert mdm == MDMLevel.HIGH
    
    def test_mdm_two_out_of_three_rule(self):
        """Test that MDM follows 2 out of 3 rule correctly"""
        calculator = MDMCalculator()
        # 2 moderate, 1 low -> should be moderate
        elements = MDMElements(
            problem_complexity=ProblemComplexity.MODERATE,
            data_complexity=DataComplexity.MODERATE,
            risk_level=RiskLevel.LOW
        )
        mdm = calculator.calculate_mdm_level(elements)
        assert mdm == MDMLevel.MODERATE


class TestProblemComplexityHelpers:
    """Test problem complexity helper functions"""
    
    def test_minimal_problem(self):
        """Test minimal problem complexity"""
        complexity = determine_problem_complexity(num_problems=1)
        assert complexity == ProblemComplexity.MINIMAL
    
    def test_low_problem_multiple_minor(self):
        """Test low complexity with multiple minor problems"""
        complexity = determine_problem_complexity(num_problems=2)
        assert complexity == ProblemComplexity.LOW
    
    def test_low_problem_stable_chronic(self):
        """Test low complexity with stable chronic illness"""
        complexity = determine_problem_complexity(
            num_problems=1,
            has_chronic_illness=True
        )
        assert complexity == ProblemComplexity.LOW
    
    def test_moderate_problem_exacerbation(self):
        """Test moderate complexity with chronic illness exacerbation"""
        complexity = determine_problem_complexity(
            num_problems=1,
            chronic_illness_exacerbation=True
        )
        assert complexity == ProblemComplexity.MODERATE
    
    def test_high_problem_life_threatening(self):
        """Test high complexity with life-threatening condition"""
        complexity = determine_problem_complexity(
            num_problems=1,
            life_threatening_condition=True
        )
        assert complexity == ProblemComplexity.HIGH


class TestDataComplexityHelpers:
    """Test data complexity helper functions"""
    
    def test_minimal_data(self):
        """Test minimal data complexity"""
        complexity = determine_data_complexity()
        assert complexity == DataComplexity.MINIMAL_NONE
    
    def test_limited_data(self):
        """Test limited data complexity"""
        complexity = determine_data_complexity(reviewed_external_records=True)
        assert complexity == DataComplexity.LIMITED
    
    def test_moderate_data(self):
        """Test moderate data complexity"""
        complexity = determine_data_complexity(independent_interpretation=True)
        assert complexity == DataComplexity.MODERATE
    
    def test_extensive_data(self):
        """Test extensive data complexity"""
        complexity = determine_data_complexity(
            independent_interpretation=True,
            discussed_with_external_physician=True
        )
        assert complexity == DataComplexity.EXTENSIVE


class TestRiskLevelHelpers:
    """Test risk level helper functions"""
    
    def test_minimal_risk(self):
        """Test minimal risk level"""
        risk = determine_risk_level()
        assert risk == RiskLevel.MINIMAL
    
    def test_low_risk(self):
        """Test low risk level"""
        risk = determine_risk_level(prescription_drug_management=True)
        assert risk == RiskLevel.LOW
    
    def test_moderate_risk(self):
        """Test moderate risk level"""
        risk = determine_risk_level(decision_for_surgery=True)
        assert risk == RiskLevel.MODERATE
    
    def test_high_risk(self):
        """Test high risk level"""
        risk = determine_risk_level(acute_threat_to_life=True)
        assert risk == RiskLevel.HIGH


class TestEMCodeSelector:
    """Test E&M code selection logic"""
    
    def test_select_by_mdm_only(self):
        """Test code selection using only MDM"""
        elements = MDMElements(
            problem_complexity=ProblemComplexity.MODERATE,
            data_complexity=DataComplexity.MODERATE,
            risk_level=RiskLevel.MODERATE
        )
        encounter = EMEncounterData(
            patient_type=PatientType.NEW,
            mdm_elements=elements
        )
        
        selector = EMCodeSelector()
        result = selector.select_code(encounter)
        
        assert result.selected_code.code == "99204"
        assert result.selection_method == "mdm"
        assert result.mdm_level == MDMLevel.MODERATE
    
    def test_select_by_time_only(self):
        """Test code selection using only time"""
        encounter = EMEncounterData(
            patient_type=PatientType.ESTABLISHED,
            total_time_minutes=35
        )
        
        selector = EMCodeSelector()
        result = selector.select_code(encounter)
        
        assert result.selected_code.code == "99214"
        assert result.selection_method == "time"
    
    def test_select_higher_code_when_both_available(self):
        """Test that higher code is selected when both MDM and time available"""
        # MDM suggests 99203 (low), time suggests 99204 (moderate)
        elements = MDMElements(
            problem_complexity=ProblemComplexity.LOW,
            data_complexity=DataComplexity.LIMITED,
            risk_level=RiskLevel.LOW
        )
        encounter = EMEncounterData(
            patient_type=PatientType.NEW,
            mdm_elements=elements,
            total_time_minutes=50  # 99204 time range
        )
        
        selector = EMCodeSelector()
        result = selector.select_code(encounter)
        
        assert result.selected_code.code == "99204"
        assert result.selection_method == "time"
    
    def test_established_patient_low_complexity(self):
        """Test established patient with low complexity"""
        elements = MDMElements(
            problem_complexity=ProblemComplexity.LOW,
            data_complexity=DataComplexity.LIMITED,
            risk_level=RiskLevel.LOW
        )
        encounter = EMEncounterData(
            patient_type=PatientType.ESTABLISHED,
            mdm_elements=elements
        )
        
        selector = EMCodeSelector()
        result = selector.select_code(encounter)
        
        assert result.selected_code.code == "99213"
        assert result.mdm_level == MDMLevel.LOW
    
    def test_invalid_encounter_raises_error(self):
        """Test that invalid encounter data raises ValueError"""
        encounter = EMEncounterData(patient_type=PatientType.NEW)
        
        selector = EMCodeSelector()
        with pytest.raises(ValueError, match="Encounter must include either MDM elements or total time"):
            selector.select_code(encounter)
    
    def test_confidence_high_when_both_available(self):
        """Test confidence is high when both MDM and time are available"""
        elements = MDMElements(
            problem_complexity=ProblemComplexity.MODERATE,
            data_complexity=DataComplexity.MODERATE,
            risk_level=RiskLevel.MODERATE
        )
        encounter = EMEncounterData(
            patient_type=PatientType.NEW,
            mdm_elements=elements,
            total_time_minutes=50
        )
        
        selector = EMCodeSelector()
        result = selector.select_code(encounter)
        
        assert result.confidence == "high"


class TestClinicalScenarios:
    """Test with realistic clinical scenarios"""
    
    def test_simple_cold_visit(self):
        """Test: Established patient with simple cold"""
        # Simple cold: minimal problem, no data review, low risk (OTC recommendations)
        result = select_em_code_simple(
            patient_type=PatientType.ESTABLISHED,
            mdm_elements=MDMElements(
                problem_complexity=ProblemComplexity.MINIMAL,
                data_complexity=DataComplexity.MINIMAL_NONE,
                risk_level=RiskLevel.MINIMAL
            ),
            time_minutes=10
        )
        
        assert result.selected_code.code in ["99211", "99212"]
    
    def test_diabetes_followup(self):
        """Test: Established patient with stable diabetes follow-up"""
        # Stable chronic illness, reviewed labs, prescription management
        result = select_em_code_simple(
            patient_type=PatientType.ESTABLISHED,
            mdm_elements=MDMElements(
                problem_complexity=ProblemComplexity.LOW,
                data_complexity=DataComplexity.LIMITED,
                risk_level=RiskLevel.LOW
            ),
            time_minutes=25
        )
        
        assert result.selected_code.code == "99213"
    
    def test_acute_chest_pain(self):
        """Test: New patient with acute chest pain"""
        # Potentially life-threatening, extensive workup, high risk
        result = select_em_code_simple(
            patient_type=PatientType.NEW,
            mdm_elements=MDMElements(
                problem_complexity=ProblemComplexity.HIGH,
                data_complexity=DataComplexity.EXTENSIVE,
                risk_level=RiskLevel.HIGH
            ),
            time_minutes=75
        )
        
        assert result.selected_code.code == "99205"
        assert result.mdm_level == MDMLevel.HIGH
    
    def test_hypertension_with_exacerbation(self):
        """Test: Established patient with hypertension exacerbation"""
        # Chronic illness with exacerbation, moderate risk
        result = select_em_code_simple(
            patient_type=PatientType.ESTABLISHED,
            mdm_elements=MDMElements(
                problem_complexity=ProblemComplexity.MODERATE,
                data_complexity=DataComplexity.MODERATE,
                risk_level=RiskLevel.MODERATE
            ),
            time_minutes=35
        )
        
        assert result.selected_code.code == "99214"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
