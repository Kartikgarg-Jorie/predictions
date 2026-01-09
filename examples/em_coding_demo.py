"""
E&M Coding Demo - Realistic Clinical Scenarios

This demo shows how to use the E&M coding system with real-world examples.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.em_codes import PatientType
from src.core.em_mdm_calculator import (
    MDMElements,
    determine_problem_complexity,
    determine_data_complexity,
    determine_risk_level
)
from src.core.em_selector import select_em_code_simple


def print_scenario(title: str, result):
    """Print formatted scenario result"""
    print("\n" + "="*70)
    print(f"📋 SCENARIO: {title}")
    print("="*70)
    print(f"Selected Code: {result.selected_code.code}")
    print(f"Description: {result.selected_code.description}")
    print(f"Selection Method: {result.selection_method.upper()}")
    if result.mdm_level:
        print(f"MDM Level: {result.mdm_level.value}")
    if result.time_minutes:
        print(f"Time: {result.time_minutes} minutes")
    print(f"Confidence: {result.confidence.upper()}")
    print(f"\nRationale: {result.rationale}")
    print("="*70)


def scenario_1_simple_uri():
    """Scenario 1: Established patient with simple URI"""
    print("\n🏥 Scenario 1: Simple Upper Respiratory Infection")
    print("Patient: Established patient, 35-year-old")
    print("Chief Complaint: Runny nose, sore throat for 3 days")
    print("Assessment: Viral URI, no complications")
    print("Plan: Symptomatic treatment, OTC recommendations")
    
    # Determine MDM elements
    problem = determine_problem_complexity(
        num_problems=1  # Single minor problem
    )
    data = determine_data_complexity()  # No external data review
    risk = determine_risk_level()  # No prescriptions needed
    
    result = select_em_code_simple(
        patient_type=PatientType.ESTABLISHED,
        mdm_elements=MDMElements(problem, data, risk),
        time_minutes=12
    )
    
    print_scenario("Simple URI Visit", result)
    return result


def scenario_2_diabetes_management():
    """Scenario 2: Diabetes follow-up with complications"""
    print("\n🏥 Scenario 2: Diabetes Type 2 Follow-up")
    print("Patient: Established patient, 58-year-old")
    print("Chief Complaint: Diabetes follow-up, foot numbness")
    print("Assessment: Type 2 DM with peripheral neuropathy")
    print("Data: Reviewed recent A1C, lipid panel")
    print("Plan: Adjust metformin, add gabapentin for neuropathy")
    
    problem = determine_problem_complexity(
        num_problems=2,  # Diabetes + neuropathy
        has_chronic_illness=True
    )
    data = determine_data_complexity(
        reviewed_external_records=True  # Reviewed lab results
    )
    risk = determine_risk_level(
        prescription_drug_management=True
    )
    
    result = select_em_code_simple(
        patient_type=PatientType.ESTABLISHED,
        mdm_elements=MDMElements(problem, data, risk),
        time_minutes=28
    )
    
    print_scenario("Diabetes Management", result)
    return result


def scenario_3_new_patient_hypertension():
    """Scenario 3: New patient with multiple chronic conditions"""
    print("\n🏥 Scenario 3: New Patient - Multiple Chronic Conditions")
    print("Patient: New patient, 62-year-old")
    print("Chief Complaint: Establish care, multiple medications")
    print("Assessment: HTN, hyperlipidemia, GERD - all stable")
    print("Data: Reviewed records from previous physician")
    print("Plan: Continue current medications, order baseline labs")
    
    problem = determine_problem_complexity(
        num_problems=3,
        has_chronic_illness=True  # Multiple stable chronic illnesses
    )
    data = determine_data_complexity(
        reviewed_external_records=True
    )
    risk = determine_risk_level(
        prescription_drug_management=True
    )
    
    result = select_em_code_simple(
        patient_type=PatientType.NEW,
        mdm_elements=MDMElements(problem, data, risk),
        time_minutes=42
    )
    
    print_scenario("New Patient Comprehensive Visit", result)
    return result


def scenario_4_copd_exacerbation():
    """Scenario 4: COPD exacerbation requiring intensive management"""
    print("\n🏥 Scenario 4: COPD Exacerbation")
    print("Patient: Established patient, 71-year-old")
    print("Chief Complaint: Increased shortness of breath, productive cough")
    print("Assessment: COPD exacerbation with acute bronchitis")
    print("Data: Reviewed CXR, discussed with pulmonologist")
    print("Plan: Prednisone taper, antibiotics, increase bronchodilators")
    
    problem = determine_problem_complexity(
        num_problems=2,
        has_chronic_illness=True,
        chronic_illness_exacerbation=True
    )
    data = determine_data_complexity(
        independent_interpretation=True,  # Interpreted CXR
        discussed_with_external_physician=True
    )
    risk = determine_risk_level(
        drug_therapy_requiring_monitoring=True  # Prednisone requires monitoring
    )
    
    result = select_em_code_simple(
        patient_type=PatientType.ESTABLISHED,
        mdm_elements=MDMElements(problem, data, risk),
        time_minutes=38
    )
    
    print_scenario("COPD Exacerbation", result)
    return result


def scenario_5_acute_chest_pain():
    """Scenario 5: New patient with acute chest pain"""
    print("\n🏥 Scenario 5: Acute Chest Pain Evaluation")
    print("Patient: New patient, 55-year-old with cardiac risk factors")
    print("Chief Complaint: Chest pain for 2 hours")
    print("Assessment: Rule out acute coronary syndrome")
    print("Data: ECG performed and interpreted, reviewed old records")
    print("Plan: Urgent cardiology referral, aspirin, nitro if needed")
    
    problem = determine_problem_complexity(
        num_problems=1,
        life_threatening_condition=True,  # Potential ACS
        undiagnosed_uncertain_prognosis=True
    )
    data = determine_data_complexity(
        independent_interpretation=True,  # ECG interpretation
        extensive_record_review=True
    )
    risk = determine_risk_level(
        acute_threat_to_life=True
    )
    
    result = select_em_code_simple(
        patient_type=PatientType.NEW,
        mdm_elements=MDMElements(problem, data, risk),
        time_minutes=82
    )
    
    print_scenario("Acute Chest Pain", result)
    return result


def scenario_6_time_based_selection():
    """Scenario 6: Using time-based selection"""
    print("\n🏥 Scenario 6: Prolonged Visit (Time-Based)")
    print("Patient: Established patient, 45-year-old")
    print("Chief Complaint: Depression counseling, medication adjustment")
    print("Time: Extensive counseling regarding treatment options")
    print("Total Time: 52 minutes on date of encounter")
    
    # Also provide MDM for comparison
    problem = determine_problem_complexity(
        num_problems=1,
        has_chronic_illness=True
    )
    data = determine_data_complexity()
    risk = determine_risk_level(
        prescription_drug_management=True
    )
    
    result = select_em_code_simple(
        patient_type=PatientType.ESTABLISHED,
        mdm_elements=MDMElements(problem, data, risk),
        time_minutes=52  # This will drive code selection to higher level
    )
    
    print_scenario("Prolonged Counseling Visit", result)
    return result


def main():
    """Run all demo scenarios"""
    print("\n" + "🏥"*35)
    print("E&M CODING DEMONSTRATION")
    print("Medical Coding Automation - Office Visit Codes")
    print("Based on CPT 2024 Guidelines")
    print("🏥"*35)
    
    # Run all scenarios
    results = []
    results.append(scenario_1_simple_uri())
    results.append(scenario_2_diabetes_management())
    results.append(scenario_3_new_patient_hypertension())
    results.append(scenario_4_copd_exacerbation())
    results.append(scenario_5_acute_chest_pain())
    results.append(scenario_6_time_based_selection())
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY OF SELECTED CODES")
    print("="*70)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.selected_code.code} - {result.selected_code.description.split(',')[1]}")
    print("="*70)
    
    print("\n✅ Demo completed successfully!")
    print("💡 All code selections follow official CPT 2024 guidelines")


if __name__ == "__main__":
    main()
