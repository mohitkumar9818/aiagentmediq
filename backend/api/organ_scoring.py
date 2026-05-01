"""
MediQ Organ System Scoring Engine (v6.0)
Maps biomarkers to organ systems and calculates accurate health scores.

ORGAN SYSTEM MAPPINGS:
- Cardiac:      Total Cholesterol, HDL, LDL, Triglycerides
- Hematologic:  Hemoglobin, WBC, RBC, Platelets
- Hepatic:      SGPT (ALT), SGOT (AST)
- Renal:        Creatinine, Urea
- Metabolic:    Glucose (Fasting), Cholesterol, Triglycerides, HDL, LDL
"""

import statistics
from typing import List, Dict, Tuple

# ======================================================
# REFERENCE RANGES (CLINICAL STANDARDS)
# ======================================================

REFERENCE_RANGES = {
    # Hematologic
    "hemoglobin": {"low": 12, "high": 17.5, "unit": "g/dL", "organ": "hematologic"},
    "wbc": {"low": 4.5, "high": 11, "unit": "K/uL", "organ": "hematologic"},
    "rbc": {"low": 4, "high": 5.5, "unit": "M/uL", "organ": "hematologic"},
    "platelets": {"low": 150, "high": 400, "unit": "K/uL", "organ": "hematologic"},
    
    # Metabolic
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL", "organ": "metabolic"},
    "fasting glucose": {"low": 70, "high": 100, "unit": "mg/dL", "organ": "metabolic"},
    
    # Cardiac & Lipid
    "total cholesterol": {"low": 0, "high": 200, "unit": "mg/dL", "organ": "cardiac"},
    "cholesterol": {"low": 0, "high": 200, "unit": "mg/dL", "organ": "cardiac"},
    "hdl": {"low": 40, "high": 300, "unit": "mg/dL", "organ": "cardiac"},
    "ldl": {"low": 0, "high": 100, "unit": "mg/dL", "organ": "cardiac"},
    "triglycerides": {"low": 0, "high": 150, "unit": "mg/dL", "organ": "cardiac"},
    
    # Hepatic
    "sgpt": {"low": 7, "high": 35, "unit": "U/L", "organ": "hepatic"},
    "alt": {"low": 7, "high": 35, "unit": "U/L", "organ": "hepatic"},
    "sgot": {"low": 10, "high": 40, "unit": "U/L", "organ": "hepatic"},
    "ast": {"low": 10, "high": 40, "unit": "U/L", "organ": "hepatic"},
    
    # Renal
    "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL", "organ": "renal"},
    "urea": {"low": 7, "high": 20, "unit": "mg/dL", "organ": "renal"},
    "bun": {"low": 7, "high": 20, "unit": "mg/dL", "organ": "renal"},
}

ORGAN_SYSTEMS = {
    "cardiac": "Cardiac",
    "hematologic": "Hematologic",
    "hepatic": "Hepatic",
    "metabolic": "Metabolic",
    "renal": "Renal"
}


# ======================================================
# SCORING LOGIC
# ======================================================

def assess_parameter_status(value: float, param_name: str) -> Tuple[str, float]:
    """
    Assess a single parameter value and assign status + score.
    
    Returns:
        (status, score) where:
        - status: "normal" | "low" | "high" | "critical"
        - score: 0-100 (100 for normal, 70-90 for slightly abnormal, <50 for critical)
    """
    param_key = param_name.lower().strip()
    
    # Try exact match first
    if param_key not in REFERENCE_RANGES:
        # Try partial match
        for key in REFERENCE_RANGES.keys():
            if key in param_key or param_key in key:
                param_key = key
                break
        else:
            # No match found - assume normal
            return "normal", 85
    
    ref = REFERENCE_RANGES[param_key]
    low, high = ref["low"], ref["high"]
    
    # Determine status and calculate score
    if low <= value <= high:
        # NORMAL: Return high score (90-100)
        return "normal", 95
    elif value < low:
        # LOW: Assess how far below normal
        if low == 0:  # Prevent division by zero
            deviation = abs(value) * 100
        else:
            deviation = (low - value) / low * 100
        
        if deviation > 50:  # More than 50% below normal
            return "critical", 30
        elif deviation > 20:  # 20-50% below normal
            return "low", 65
        else:  # 0-20% below normal (slightly low)
            return "low", 80
    else:  # value > high
        # HIGH: Assess how far above normal
        if high == 0:  # Prevent division by zero
            deviation = value * 100
        else:
            deviation = (value - high) / high * 100
        
        if deviation > 100:  # More than 100% above normal
            return "critical", 25
        elif deviation > 40:  # 40-100% above normal
            return "high", 60
        else:  # 0-40% above normal (slightly high)
            return "high", 75
    

def get_parameter_status(value: float, param_name: str) -> str:
    """Get only the status (normal/low/high/critical) for a parameter."""
    status, _ = assess_parameter_status(value, param_name)
    return status


def get_parameter_score(value: float, param_name: str) -> float:
    """Get only the score (0-100) for a parameter."""
    _, score = assess_parameter_status(value, param_name)
    return score


def validate_parameter_value(value) -> Tuple[bool, float | None]:
    """
    Validate and convert a parameter value to float.
    
    Returns:
        (is_valid, numeric_value)
    """
    try:
        val = float(str(value).strip())
        # Allow up to 10,000,000 for values like Platelets (e.g., 400000)
        if 0 <= val <= 10000000:  # Only positive medical values
            return True, val
        return False, None
    except (ValueError, TypeError, AttributeError):
        return False, None


def calculate_organ_scores(parameters: List[Dict]) -> Dict[str, int]:
    """
    Calculate organ system scores from biomarker parameters.
    
    Args:
        parameters: List of parameter dicts with keys: name, value, status
        
    Returns:
        Dictionary with organ scores (0-100 for each organ system)
    """
    if not parameters:
        return {
            "cardiac": 0,
            "hematologic": 0,
            "hepatic": 0,
            "metabolic": 0,
            "renal": 0
        }
    
    organ_scores_map = {
        "cardiac": [],
        "hematologic": [],
        "hepatic": [],
        "metabolic": [],
        "renal": []
    }
    
    # Group parameters by organ system
    for param in parameters:
        if not isinstance(param, dict):
            continue
            
        param_name = str(param.get("name", "")).lower().strip()
        value = param.get("value")
        
        # Validate numeric value
        is_valid, numeric_val = validate_parameter_value(value)
        if not is_valid:
            continue
        
        # Find matching reference range
        matched_key = None
        for ref_key in REFERENCE_RANGES.keys():
            if ref_key in param_name or param_name in ref_key:
                matched_key = ref_key
                break
        
        if not matched_key:
            continue
        
        organ = REFERENCE_RANGES[matched_key]["organ"]
        status = str(param.get("status", "normal")).lower()
        if status == "normal":
            score = 95
        elif status in ["low", "high"]:
            score = 75
        elif status == "critical":
            score = 30
        else:
            score = 85
        organ_scores_map[organ].append(score)
    
    # Calculate average for each organ
    result = {}
    for organ, scores in organ_scores_map.items():
        if scores:
            result[organ] = max(0, min(100, int(statistics.mean(scores))))
        else:
            result[organ] = 0
    
    return result


def calculate_health_score(parameters: List[Dict]) -> Dict:
    """
    Calculate overall health score from all parameters.
    
    Args:
        parameters: List of parameter dicts with keys: name, value, status
        
    Returns:
        Dictionary with: health_score, overall_risk, critical_count, abnormal_count
    """
    if not parameters:
        return {
            "health_score": 0,
            "overall_risk": "unknown",
            "critical_count": 0,
            "abnormal_count": 0,
            "normal_count": 0,
            "average_confidence": 0
        }
    
    scores = []
    critical_count = 0
    abnormal_count = 0
    normal_count = 0
    
    for param in parameters:
        if not isinstance(param, dict):
            continue
        
        param_name = str(param.get("name", "")).lower().strip()
        value = param.get("value")
        status = str(param.get("status", "normal")).lower().strip()
        
        # Validate numeric value
        is_valid, numeric_val = validate_parameter_value(value)
        if not is_valid:
            continue
        
        # Get score based on status
        if status == "normal":
            score = 95
        elif status in ["low", "high"]:
            score = 75
        elif status == "critical":
            score = 30
        else:
            score = 85
        scores.append(score)
        
        # Count statuses
        if status == "critical":
            critical_count += 1
        elif status in ["high", "low"]:
            abnormal_count += 1
        elif status == "normal":
            normal_count += 1
    
    if not scores:
        return {
            "health_score": 0,
            "overall_risk": "unknown",
            "critical_count": 0,
            "abnormal_count": 0,
            "normal_count": 0,
            "average_confidence": 0
        }
    
    # Calculate health score as average of all parameter scores
    health_score = int(statistics.mean(scores))
    health_score = max(0, min(100, health_score))
    
    # Determine risk level
    if health_score >= 75:
        overall_risk = "low-risk"
    elif health_score >= 50:
        overall_risk = "moderate-risk"
    else:
        overall_risk = "high-risk"
    
    return {
        "health_score": health_score,
        "overall_risk": overall_risk,
        "critical_count": critical_count,
        "abnormal_count": abnormal_count,
        "normal_count": normal_count,
        "average_confidence": 0.85  # Default confidence
    }


def enrich_parameters(parameters: List[Dict]) -> List[Dict]:
    """
    Enrich parameter data with accurate status and scoring.
    
    This function evaluates each parameter based on its provided normalRange
    instead of strictly relying on hardcoded reference values.
    
    Args:
        parameters: List of parameter dicts
        
    Returns:
        Enriched parameters with accurate status
    """
    if not parameters:
        return []
    
    import re
    enriched = []
    for param in parameters:
        if not isinstance(param, dict):
            continue
        
        param_name = str(param.get("name", "")).lower().strip()
        value = param.get("value")
        ref_range = str(param.get("normalRange", "")).strip()
        unit_lower = str(param.get("unit", "")).lower()
        
        # Validate numeric value
        is_valid, numeric_val = validate_parameter_value(value)
        if not is_valid:
            enriched.append(param)  # Keep original if can't parse
            continue
        
        status = None
        
        # Check against normalRange if available
        if ref_range:
            try:
                compare_val = numeric_val
                if "-" in ref_range:
                    parts = ref_range.split("-")
                    min_val = float(re.search(r"(\d+\.?\d*)", parts[0]).group(1))
                    max_val = float(re.search(r"(\d+\.?\d*)", parts[1]).group(1))
                    
                    # Scale adjustment for lakhs/millions/thousands
                    if "lakh" in unit_lower and compare_val < 1000 and min_val >= 1000:
                        compare_val *= 100000
                    elif "million" in unit_lower and compare_val < 100 and min_val >= 1000:
                        compare_val *= 1000000
                    elif ("k" in unit_lower or "thou" in unit_lower) and compare_val < 1000 and min_val >= 1000:
                        compare_val *= 1000
                        
                    if compare_val < min_val or compare_val > max_val:
                        status = "critical"
                    else:
                        status = "normal"
                elif "<" in ref_range:
                    max_val = float(re.search(r"(\d+\.?\d*)", ref_range).group(1))
                    if compare_val > max_val:
                        status = "critical"
                    else:
                        status = "normal"
                elif ">" in ref_range:
                    min_val = float(re.search(r"(\d+\.?\d*)", ref_range).group(1))
                    if compare_val < min_val:
                        status = "critical"
                    else:
                        status = "normal"
            except Exception:
                pass
        
        # Fallback if no valid range could be parsed
        if not status:
            status, _ = assess_parameter_status(numeric_val, param_name)
        
        # Update parameter
        enriched_param = param.copy()
        enriched_param["status"] = status
        enriched_param["confidence"] = 0.92  # High confidence
        enriched.append(enriched_param)
    
    return enriched
