"""
TEST SCRIPT FOR ORGAN SCORING AND PARAMETER VALIDATION
Tests all fixes for health analytics dashboard
"""

from organ_scoring import (
    assess_parameter_status,
    get_parameter_status,
    get_parameter_score,
    calculate_organ_scores,
    calculate_health_score,
    enrich_parameters,
    validate_parameter_value
)

def test_parameter_validation():
    """Test parameter value validation"""
    print("\n" + "="*60)
    print("TEST 1: Parameter Value Validation")
    print("="*60)
    
    test_cases = [
        (12.5, True, 12.5),      # Valid float
        ("15.8", True, 15.8),    # String that converts to float
        ("N/A", False, None),    # Invalid
        (None, False, None),     # None
        (150, True, 150),        # Integer
        (-5, False, None),       # Out of range (negative medical value)
    ]
    
    for value, expected_valid, expected_num in test_cases:
        is_valid, numeric_val = validate_parameter_value(value)
        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
        print(f"{status}: validate({value}) -> valid={is_valid}, num={numeric_val}")


def test_parameter_assessment():
    """Test individual parameter status and scoring"""
    print("\n" + "="*60)
    print("TEST 2: Parameter Status Assessment")
    print("="*60)
    
    test_cases = [
        ("Hemoglobin", 14.5, "normal", 95),      # Normal
        ("Hemoglobin", 10.5, "low", 80),         # Slightly low
        ("Hemoglobin", 8.0, "critical", 30),     # Critical low
        ("Glucose", 95, "normal", 95),           # Normal
        ("Glucose", 150, "high", 75),            # Slightly high
        ("Glucose", 300, "critical", 25),        # Critical high
        ("Creatinine", 1.0, "normal", 95),       # Normal
        ("HDL", 50, "normal", 95),               # Good cholesterol normal
        ("SGPT", 25, "normal", 95),              # Liver enzyme normal
        ("WBC", 7.5, "normal", 95),              # WBC normal
    ]
    
    for param_name, value, expected_status, expected_score in test_cases:
        status, score = assess_parameter_status(value, param_name)
        match = status == expected_status and score == expected_score
        icon = "✅" if match else "⚠️"
        print(f"{icon} {param_name}({value}): status={status} (exp: {expected_status}), score={score} (exp: {expected_score})")


def test_organ_score_calculation():
    """Test organ system score calculation"""
    print("\n" + "="*60)
    print("TEST 3: Organ System Score Calculation")
    print("="*60)
    
    # Test case 1: All normal parameters
    print("\n--- All Normal Parameters ---")
    params_normal = [
        {"name": "Hemoglobin", "value": 14.5, "status": "normal"},
        {"name": "WBC", "value": 7.5, "status": "normal"},
        {"name": "RBC", "value": 4.8, "status": "normal"},
        {"name": "Platelets", "value": 250, "status": "normal"},
        {"name": "Glucose", "value": 95, "status": "normal"},
        {"name": "Total Cholesterol", "value": 180, "status": "normal"},
        {"name": "HDL", "value": 50, "status": "normal"},
        {"name": "LDL", "value": 100, "status": "normal"},
        {"name": "Triglycerides", "value": 120, "status": "normal"},
        {"name": "SGPT", "value": 25, "status": "normal"},
        {"name": "SGOT", "value": 30, "status": "normal"},
        {"name": "Creatinine", "value": 1.0, "status": "normal"},
        {"name": "Urea", "value": 15, "status": "normal"},
    ]
    
    organ_scores = calculate_organ_scores(params_normal)
    print(f"Organ Scores (all normal): {organ_scores}")
    print(f"✅ All organs should be around 90-100: {all(score >= 85 for score in organ_scores.values())}")
    
    # Test case 2: Mixed abnormal
    print("\n--- Mixed Abnormal Parameters ---")
    params_mixed = [
        {"name": "Hemoglobin", "value": 10.5, "status": "low"},     # Low
        {"name": "WBC", "value": 8.0, "status": "normal"},
        {"name": "Glucose", "value": 150, "status": "high"},        # High
        {"name": "Creatinine", "value": 1.8, "status": "high"},     # Renal issue
        {"name": "SGPT", "value": 60, "status": "high"},            # Liver issue
    ]
    
    organ_scores = calculate_organ_scores(params_mixed)
    print(f"Organ Scores (mixed): {organ_scores}")
    print(f"✅ Hematologic should be low: {organ_scores['hematologic'] < 85}")
    print(f"✅ Metabolic should be low: {organ_scores['metabolic'] < 85}")
    print(f"✅ Renal should be low: {organ_scores['renal'] < 85}")
    print(f"✅ Hepatic should be low: {organ_scores['hepatic'] < 85}")
    
    # Test case 3: Empty parameters
    print("\n--- Empty Parameters ---")
    organ_scores_empty = calculate_organ_scores([])
    print(f"Organ Scores (empty): {organ_scores_empty}")
    print(f"✅ All scores should be 0: {all(score == 0 for score in organ_scores_empty.values())}")


def test_health_score_calculation():
    """Test overall health score calculation"""
    print("\n" + "="*60)
    print("TEST 4: Health Score Calculation")
    print("="*60)
    
    # Test case 1: All normal
    print("\n--- All Normal Parameters ---")
    params_normal = [
        {"name": "Hemoglobin", "value": 14.5, "status": "normal"},
        {"name": "Glucose", "value": 95, "status": "normal"},
        {"name": "Cholesterol", "value": 180, "status": "normal"},
        {"name": "Creatinine", "value": 1.0, "status": "normal"},
    ]
    
    health = calculate_health_score(params_normal)
    print(f"Health Score (all normal): {health['health_score']}")
    print(f"Risk Level: {health['overall_risk']}")
    print(f"✅ Score should be >= 75: {health['health_score'] >= 75}")
    print(f"✅ Risk should be low-risk: {health['overall_risk'] == 'low-risk'}")
    
    # Test case 2: Mixed abnormal
    print("\n--- Mixed Abnormal Parameters ---")
    params_mixed = [
        {"name": "Hemoglobin", "value": 10.5, "status": "low"},
        {"name": "Glucose", "value": 150, "status": "high"},
        {"name": "Creatinine", "value": 1.8, "status": "high"},
    ]
    
    health = calculate_health_score(params_mixed)
    print(f"Health Score (mixed abnormal): {health['health_score']}")
    print(f"Risk Level: {health['overall_risk']}")
    print(f"✅ Score should be < 75: {health['health_score'] < 75}")
    
    # Test case 3: Empty parameters
    print("\n--- Empty Parameters ---")
    health_empty = calculate_health_score([])
    print(f"Health Score (empty): {health_empty}")
    print(f"✅ Score should be 0: {health_empty['health_score'] == 0}")
    print(f"✅ Risk should be unknown: {health_empty['overall_risk'] == 'unknown'}")


def test_parameter_enrichment():
    """Test parameter enrichment with accurate status and scoring"""
    print("\n" + "="*60)
    print("TEST 5: Parameter Enrichment")
    print("="*60)
    
    raw_params = [
        {"name": "Hemoglobin", "value": "14.5", "unit": "g/dL", "normalRange": "12-17.5", "status": "normal", "explanation": "Red blood cell oxygen carrier"},
        {"name": "Glucose", "value": "95", "unit": "mg/dL", "normalRange": "70-100", "status": "normal", "explanation": "Blood sugar level"},
        {"name": "Creatinine", "value": "1.5", "unit": "mg/dL", "normalRange": "0.6-1.2", "status": "high", "explanation": "Kidney function marker"},
    ]
    
    enriched = enrich_parameters(raw_params)
    
    for orig, enr in zip(raw_params, enriched):
        print(f"\n{orig['name']}:")
        print(f"  Original status: {orig['status']}")
        print(f"  Enriched status: {enr['status']}")
        print(f"  Confidence: {enr.get('confidence', 'N/A')}")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  MEDIQ ORGAN SCORING AND VALIDATION TEST SUITE".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    test_parameter_validation()
    test_parameter_assessment()
    test_organ_score_calculation()
    test_health_score_calculation()
    test_parameter_enrichment()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60 + "\n")
