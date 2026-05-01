"""
Robust Biomarker Parser for Medical Reports
Extracts lab values directly from PDF text using pattern matching and regex.
Handles table formats, various units, and reference ranges.
"""

import re
from typing import List, Dict, Optional, Tuple

# ======================================================
# BIOMARKER REFERENCE DATA
# ======================================================

BIOMARKER_REFERENCES = {
    "hemoglobin": {
        "units": ["g/dl", "g/dL", "g/L", "gm/dl", "gm%"],
        "range_male": "13.0 - 17.0",
        "range_female": "12.0 - 15.5",
        "aliases": ["hgb", "hb", "hemoglobin"],
        "organ": "hematologic"
    },
    "wbc": {
        "units": ["/µl", "/µL", "/uL", "/ul", "cells/µl", "k/µl", "thou/µl"],
        "range": "4500 - 11000",
        "aliases": ["wbc", "white blood cells", "wbc count", "total wbc"],
        "organ": "hematologic"
    },
    "rbc": {
        "units": ["million/µl", "million/µL", "m/µl", "m/µL", "cells/µl"],
        "range": "4.5 - 5.9",
        "aliases": ["rbc", "red blood cells", "rbc count"],
        "organ": "hematologic"
    },
    "platelets": {
        "units": ["/µl", "/µL", "/uL", "lakh/µl", "k/µl", "thou/µl"],
        "range": "150000 - 400000",
        "aliases": ["platelets", "plt", "platelet count"],
        "organ": "hematologic"
    },
    "glucose": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range_fasting": "70 - 100",
        "range_random": "< 140",
        "aliases": ["glucose", "blood glucose", "fbs", "fasting blood glucose", "random blood glucose"],
        "organ": "metabolic"
    },
    "cholesterol": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "< 200",
        "aliases": ["cholesterol", "total cholesterol", "t. cholesterol"],
        "organ": "cardiac"
    },
    "hdl": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "> 40",
        "aliases": ["hdl", "hdl cholesterol", "good cholesterol"],
        "organ": "cardiac"
    },
    "ldl": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "< 130",
        "aliases": ["ldl", "ldl cholesterol", "bad cholesterol"],
        "organ": "cardiac"
    },
    "triglycerides": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "< 150",
        "aliases": ["triglycerides", "triglyceride"],
        "organ": "cardiac"
    },
    "sgpt": {
        "units": ["u/l", "u/L", "iu/l", "iu/L", "unit/l"],
        "range": "7 - 56",
        "aliases": ["sgpt", "alt", "alanine transaminase"],
        "organ": "hepatic"
    },
    "sgot": {
        "units": ["u/l", "u/L", "iu/l", "iu/L", "unit/l"],
        "range": "10 - 40",
        "aliases": ["sgot", "ast", "aspartate transaminase"],
        "organ": "hepatic"
    },
    "creatinine": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range_male": "0.7 - 1.3",
        "range_female": "0.6 - 1.2",
        "aliases": ["creatinine", "serum creatinine"],
        "organ": "renal"
    },
    "urea": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "15 - 45",
        "aliases": ["urea", "blood urea", "bun", "urea nitrogen"],
        "organ": "renal"
    },
    "sodium": {
        "units": ["meq/l", "meq/L", "mmol/l", "mmol/L"],
        "range": "135 - 145",
        "aliases": ["sodium", "na", "serum sodium"],
        "organ": "metabolic"
    },
    "potassium": {
        "units": ["meq/l", "meq/L", "mmol/l", "mmol/L"],
        "range": "3.5 - 5.0",
        "aliases": ["potassium", "k", "serum potassium"],
        "organ": "metabolic"
    },
    "calcium": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "8.5 - 10.2",
        "aliases": ["calcium", "ca", "serum calcium"],
        "organ": "metabolic"
    },
    "phosphorus": {
        "units": ["mg/dl", "mg/dL", "mmol/l", "mmol/L"],
        "range": "2.5 - 4.5",
        "aliases": ["phosphorus", "phosphate"],
        "organ": "metabolic"
    },
    "protein": {
        "units": ["g/dl", "g/dL", "g/l", "g/L"],
        "range": "6.0 - 8.3",
        "aliases": ["total protein", "tp", "protein"],
        "organ": "metabolic"
    },
    "albumin": {
        "units": ["g/dl", "g/dL", "g/l", "g/L"],
        "range": "3.5 - 5.5",
        "aliases": ["albumin", "serum albumin"],
        "organ": "hepatic"
    },
}


# ======================================================
# PARSING UTILITIES
# ======================================================

def normalize_unit(unit_str: str) -> str:
    """Normalize unit strings to standard format."""
    unit_str = unit_str.lower().strip()
    # Replace variations of common units
    unit_str = unit_str.replace("µ", "µ").replace("u", "u")
    return unit_str


def extract_numeric_value(value_str: str) -> Optional[float]:
    """Extract numeric value from a string, handling various formats."""
    value_str = str(value_str).strip()
    
    # Remove common units and formatting
    value_str = re.sub(r"\s*(mg/dl|mg/dL|g/dL|g/dl|u/l|u/L|/µl|/µL|/uL|/ul|million/µl|lakh/µl|k/µl|thou/µl|meq/l|meq/L|mmol/l|mmol/L|iu/l|iu/L|unit/l|cells/µl|gm/dl|gm%)\s*", "", value_str, flags=re.IGNORECASE)
    
    # Extract first number (handles ranges like "13.0-17.0")
    match = re.search(r"(\d+\.?\d*)", value_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_range(range_str: str) -> Optional[str]:
    """Extract and normalize reference range."""
    if not range_str:
        return None
    
    range_str = str(range_str).strip()
    # Keep as-is, just clean up whitespace
    return re.sub(r"\s+", " ", range_str)


def assess_value_status(value: float, ref_range: str, biomarker_name: str) -> str:
    """Determine if a value is normal, low, high, or critical."""
    ref_range = ref_range.lower().strip()
    biomarker_lower = biomarker_name.lower()
    
    try:
        # Handle "< X" format (e.g., cholesterol < 200)
        if ref_range.startswith("<"):
            max_val = float(re.search(r"(\d+\.?\d*)", ref_range).group(1))
            if value > max_val * 1.5:
                return "critical"
            elif value > max_val:
                return "high"
            else:
                return "normal"
        
        # Handle "> X" format (e.g., HDL > 40)
        elif ref_range.startswith(">"):
            min_val = float(re.search(r"(\d+\.?\d*)", ref_range).group(1))
            if value < min_val * 0.5:
                return "critical"
            elif value < min_val:
                return "low"
            else:
                return "normal"
        
        # Handle "X - Y" format (e.g., 13.0 - 17.0)
        elif "-" in ref_range:
            parts = ref_range.split("-")
            min_val = float(re.search(r"(\d+\.?\d*)", parts[0]).group(1))
            max_val = float(re.search(r"(\d+\.?\d*)", parts[1]).group(1))
            
            if value < min_val:
                # Check if critical
                if value < min_val * 0.7:
                    return "critical"
                return "low"
            elif value > max_val:
                # Check if critical
                if value > max_val * 1.3:
                    return "critical"
                return "high"
            else:
                return "normal"
    except Exception as e:
        print(f"⚠️ Could not assess value {value} against range {ref_range}: {e}")
    
    return "normal"


def get_explanation(biomarker_name: str, value: float, status: str) -> str:
    """Generate clinical explanation for a biomarker."""
    explanations = {
        ("hemoglobin", "normal"): "Hemoglobin level is normal, indicating adequate oxygen-carrying capacity.",
        ("hemoglobin", "low"): "Low hemoglobin may indicate anemia. Dietary iron and B12 intake should be evaluated.",
        ("hemoglobin", "high"): "High hemoglobin could indicate dehydration or polycythemia. Hydration status should be checked.",
        ("hemoglobin", "critical"): "Critically low hemoglobin requires medical attention.",
        
        ("wbc", "normal"): "White blood cell count is normal, indicating adequate immune function.",
        ("wbc", "low"): "Low WBC count may indicate compromised immunity. Monitor for infections.",
        ("wbc", "high"): "High WBC count may indicate infection or inflammation.",
        ("wbc", "critical"): "Critically abnormal WBC count requires medical evaluation.",
        
        ("glucose", "normal"): "Blood glucose level is within normal range.",
        ("glucose", "low"): "Low blood glucose (hypoglycemia) requires attention. Eat easily digestible carbs.",
        ("glucose", "high"): "Elevated glucose may indicate prediabetes or diabetes. Dietary modification recommended.",
        ("glucose", "critical"): "Critically elevated glucose requires immediate medical attention.",
        
        ("cholesterol", "normal"): "Total cholesterol is at a healthy level.",
        ("cholesterol", "high"): "Elevated cholesterol increases cardiovascular risk. Diet and exercise modifications recommended.",
        ("cholesterol", "critical"): "Very high cholesterol requires medical intervention.",
        
        ("hdl", "normal"): "HDL (good cholesterol) is at a healthy protective level.",
        ("hdl", "low"): "Low HDL increases cardiovascular risk. Increase aerobic exercise and omega-3 intake.",
        
        ("ldl", "normal"): "LDL (bad cholesterol) is at a healthy level.",
        ("ldl", "high"): "Elevated LDL increases cardiovascular risk. Reduce saturated fats and increase fiber.",
        
        ("triglycerides", "normal"): "Triglyceride level is within normal range.",
        ("triglycerides", "high"): "Elevated triglycerides increase cardiovascular risk. Reduce refined carbs and alcohol.",
        
        ("creatinine", "normal"): "Creatinine level indicates normal kidney function.",
        ("creatinine", "high"): "Elevated creatinine may indicate reduced kidney function. Consult nephrologist if persistent.",
        
        ("urea", "normal"): "Urea level is normal, indicating adequate protein metabolism.",
        ("urea", "high"): "Elevated urea may indicate kidney dysfunction. Hydration and protein intake should be reviewed.",
    }
    
    key = (biomarker_name.lower(), status)
    return explanations.get(key, f"{biomarker_name} is {status}.")


# ======================================================
# CORE PARSING FUNCTION
# ======================================================

def find_biomarker_patterns(text: str) -> List[Dict]:
    """
    Search text for biomarker patterns and extract values.
    Handles formats like:
    - "Hemoglobin 14.2 g/dL (13.0 - 17.0)"
    - "WBC | 7200 /µL | 4500 - 11000"
    - "SGPT    : 32    U/L    (7 - 56)"
    - "SGPT (ALT) | 28 | U/L | (7 - 56)"
    """
    extracted = []
    text_lower = text.lower()
    
    print(f"\n📊 Starting biomarker pattern search...")
    print(f"   Text length: {len(text)} characters")
    
    for biomarker_key, info in BIOMARKER_REFERENCES.items():
        # Try each alias
        for alias in info.get("aliases", [biomarker_key]):
            # Pattern 1: "Biomarker Value Unit (Range)" - handles parenthetical names
            pattern1 = rf"\b{re.escape(alias)}(?:\s*\([^)]*\))?\s*[\:\|]?\s*(\d+\.?\d*)\s*([\w/µ°%\-\.]+)?\s*(?:\(([^)]+)\))?"
            
            # Pattern 2: "Biomarker | Value | Unit | Range"
            pattern2 = rf"\b{re.escape(alias)}\b(?:\s*\([^)]*\))?\s*\|\s*(\d+\.?\d*)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)"
            
            # Pattern 3: With dashes/spaces: "Biomarker    32    U/L    (7-56)"
            pattern3 = rf"\b{re.escape(alias)}(?:\s*\([^)]*\))?\s+(\d+\.?\d*)\s+([a-z/µ°%\-\.]+)\s+(?:\(([^)]+)\))?"
            
            # Pattern 4: Handle spaces around the value: "SGPT (ALT) | 28 | U/L"
            pattern4 = rf"\b{re.escape(alias)}\b(?:\s*\([^)]*\))?\s*[\:\|]?\s*(\d+\.?\d*)"
            
            for pattern in [pattern1, pattern2, pattern3, pattern4]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        value_str = match.group(1)
                        unit_str = match.group(2) if match.lastindex >= 2 else ""
                        range_str = match.group(3) if match.lastindex >= 3 else ""
                        
                        # Extract numeric value
                        numeric_value = extract_numeric_value(value_str)
                        if numeric_value is None:
                            continue
                        
                        # Get proper name
                        proper_name = biomarker_key.capitalize()
                        
                        # Get unit
                        unit = normalize_unit(unit_str) if unit_str else info.get("units", [""])[0]
                        
                        # Get reference range
                        ref_range = extract_range(range_str) if range_str else info.get("range", info.get("range_fasting", ""))
                        
                        # Assess status
                        status = assess_value_status(numeric_value, ref_range, biomarker_key)
                        
                        # Create parameter object
                        param = {
                            "name": proper_name,
                            "value": numeric_value,
                            "unit": unit,
                            "normalRange": ref_range,
                            "status": status,
                            "confidence": 0.95,  # High confidence for directly extracted values
                            "explanation": get_explanation(proper_name, numeric_value, status),
                            "organ": info.get("organ", "unknown")
                        }
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in extracted:
                            if existing["name"].lower() == param["name"].lower():
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            extracted.append(param)
                            print(f"   ✅ Found: {proper_name} = {numeric_value} {unit} ({status})")
                    
                    except Exception as e:
                        print(f"   ⚠️ Error processing match: {e}")
                        continue
    
    print(f"\n   📈 Total biomarkers extracted: {len(extracted)}")
    return extracted


def parse_biomarkers_from_text(text: str) -> List[Dict]:
    """
    Main entry point: Parse biomarkers from medical report text.
    Returns list of structured parameter objects.
    """
    if not text or len(text.strip()) < 50:
        print("⚠️ Text too short or empty for biomarker parsing")
        return []
    
    print("\n" + "="*60)
    print("🔬 BIOMARKER DIRECT PARSER INITIATED")
    print("="*60)
    
    biomarkers = find_biomarker_patterns(text)
    
    if biomarkers:
        print(f"\n✅ Successfully extracted {len(biomarkers)} biomarkers directly from PDF text")
    else:
        print(f"\n❌ No biomarkers found using direct parser")
    
    return biomarkers


def validate_and_merge_parameters(ai_params: List[Dict], direct_params: List[Dict]) -> List[Dict]:
    """
    Merge AI-extracted parameters with direct parser results.
    Prioritizes direct parser (regex) results, fills gaps with AI results.
    """
    merged = {}
    
    # Add direct parser results first (higher confidence)
    for param in direct_params:
        key = param.get("name", "").lower()
        if key:
            merged[key] = param
    
    # Fill in AI results for parameters we missed
    for param in ai_params:
        key = param.get("name", "").lower()
        if key and key not in merged:
            merged[key] = param
    
    print(f"\n🔀 Merged parameters: {len(direct_params)} direct + {len(ai_params)} AI = {len(merged)} total")
    
    return list(merged.values())
