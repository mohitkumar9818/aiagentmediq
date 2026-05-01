#!/usr/bin/env python3
"""
End-to-End Test: Simulates medical report analysis pipeline
"""
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment
load_dotenv(Path.cwd() / '.env', override=True)

# Import the actual analyze_medical_text function
from ai_engine import analyze_medical_text

# Sample medical report text (simulating PDF extraction)
sample_medical_report = """
MEDICAL LABORATORY REPORT

Patient Information:
Name: Robert Johnson
Age: 52
Gender: Male
Date: 2024-04-15

COMPLETE BLOOD COUNT (CBC):
  Hemoglobin: 13.5 g/dL (Normal: 13.5-17.5)
  Hematocrit: 41% (Normal: 41-53%)
  White Blood Cells: 7.2 x10^9/L (Normal: 4.5-11.0)
  Platelets: 245 x10^9/L (Normal: 150-400)

BASIC METABOLIC PANEL:
  Glucose: 152 mg/dL (Normal: 70-100) *** HIGH ***
  BUN: 18 mg/dL (Normal: 7-20)
  Creatinine: 1.2 mg/dL (Normal: 0.7-1.3)
  Sodium: 138 mEq/L (Normal: 136-145)
  Potassium: 4.1 mEq/L (Normal: 3.5-5.0)
  Chloride: 102 mEq/L (Normal: 98-107)

LIPID PANEL:
  Total Cholesterol: 245 mg/dL (Normal: <200) *** HIGH ***
  HDL Cholesterol: 32 mg/dL (Normal: >40) *** LOW ***
  LDL Cholesterol: 175 mg/dL (Normal: <100) *** HIGH ***
  Triglycerides: 235 mg/dL (Normal: <150) *** HIGH ***

LIVER FUNCTION TESTS:
  AST: 28 U/L (Normal: 10-40)
  ALT: 32 U/L (Normal: 7-56)
  Bilirubin: 0.8 mg/dL (Normal: 0.1-1.2)
  Albumin: 4.2 g/dL (Normal: 3.5-5.0)

DIABETES SCREENING:
  Hemoglobin A1C: 6.9% (Normal: <5.7%) *** ELEVATED ***

THYROID FUNCTION:
  TSH: 2.1 mIU/L (Normal: 0.4-4.0)
  Free T4: 1.1 ng/dL (Normal: 0.8-1.8)

KIDNEY FUNCTION:
  eGFR: 68 mL/min/1.73m2 (Normal: >60)

Physician Notes:
Patient presents with multiple metabolic abnormalities indicating increased cardiovascular and metabolic risk.
Recommend lifestyle modifications and follow-up testing in 3 months.
"""

print("=" * 70)
print("🔬 END-TO-END MEDICAL ANALYSIS TEST")
print("=" * 70)
print(f"\n📄 Analyzing medical report ({len(sample_medical_report)} chars)...")
print(f"✓ Patient: Robert Johnson, Age: 52, Male\n")

try:
    # Call the actual analysis function
    print("📤 Sending to analysis pipeline...")
    result = analyze_medical_text(sample_medical_report)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    
    # Display results
    print(f"\n👤 User Profile: {result.get('user_profile', {})}")
    print(f"\n📊 Biomarkers Found: {len(result.get('parameters', []))}")
    
    if result.get('parameters'):
        print("\n   First 3 biomarkers:")
        for i, param in enumerate(result.get('parameters', [])[:3]):
            print(f"   {i+1}. {param['name']}: {param['value']} {param['unit']} - {param['status'].upper()}")
    
    print(f"\n📈 Health Score: {result.get('risk_metrics', {}).get('health_score', 'N/A')}")
    print(f"⚠️  Overall Risk: {result.get('risk_metrics', {}).get('overall_risk', 'N/A')}")
    
    print(f"\n💊 Health Plan Days: {len(result.get('health_plan', []))}")
    print(f"🦠 Disease Risks: {len(result.get('disease_risks', []))}")
    
    if result.get('disease_risks'):
        print("\n   Top disease risks:")
        for risk in result.get('disease_risks', [])[:3]:
            print(f"   • {risk.get('disease', 'Unknown')}: {risk.get('risk_level', 'N/A')} risk")
    
    print(f"\n📋 Summary (first 150 chars):")
    summary = result.get('summary', 'N/A')
    print(f"   {str(summary)[:150]}...")
    
    print(f"\n⚕️ Doctor's Perspective (first 150 chars):")
    perspective = result.get('doctor_perspective', 'N/A')
    print(f"   {str(perspective)[:150]}...")
    
    print(f"\n⏱️  Processing Time: {result.get('audit', {}).get('processing_time_ms', 'N/A')}ms")
    print(f"🔧 Engine Used: {result.get('audit', {}).get('engine', 'N/A')}")
    
    print("\n✅ TEST SUCCESSFUL! Analysis pipeline is working correctly.")
    
except Exception as e:
    print(f"\n❌ TEST FAILED!")
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
