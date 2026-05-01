"""
MediQ AI Analysis Engine (v6.0)
Enhanced with: organ scores, doctor's perspective, 10-day health plan,
disease risk predictions, and prevention tips.
Dual AI support: Gemini (primary) → Groq (fallback) → Error fallback.
Direct biomarker parsing: Extracts lab values from PDF text before AI.
"""

import os
import json
import re
import hashlib
import random
import time
from datetime import datetime, timezone
from pathlib import Path as _Path
from dotenv import load_dotenv
from google import genai
from groq import Groq
from organ_scoring import (
    calculate_organ_scores,
    calculate_health_score,
    enrich_parameters,
    validate_parameter_value
)
from biomarker_parser import parse_biomarkers_from_text, validate_and_merge_parameters

# ======================================================
# ENV LOAD
# ======================================================
_ENV_PATH = _Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ======================================================
# CLIENTS & CONFIG
# ======================================================
gemini_client = None
groq_client = None
gemini_initialized = False
groq_initialized = False

# Initialize Gemini (disabled - using Groq only)
try:
    if False:  # Gemini disabled, using Groq only
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_initialized = True
        print("✅ Gemini API: Initialized")
    else:
        print("⚠️ Gemini API: Disabled (using Groq only)")
except Exception as e:
    print(f"❌ Gemini API: Failed to initialize - {e}")

# Initialize Groq
try:
    if GROQ_API_KEY and GROQ_API_KEY.strip() and not GROQ_API_KEY.startswith("your_"):
        groq_client = Groq(api_key=GROQ_API_KEY)
        groq_initialized = True
        print("✅ Groq API: Initialized (llama-3.1-8b-instant)")
    else:
        print("⚠️ Groq API: No API key provided (will not be available as fallback)")
except Exception as e:
    print(f"❌ Groq API: Failed to initialize - {e}")

# Verify at least one engine is available
if not gemini_initialized and not groq_initialized:
    print("🚨 WARNING: Neither Gemini nor Groq initialized! Add API keys to .env file.")

GEMINI_MODEL = "gemini-2.0-flash"   # ✅ stable model (replaces deprecated gemini-2.0-flash-exp)
GROQ_MODEL = "llama-3.1-8b-instant"
ENGINE_VERSION = "v6.0-enhanced-prod"

# Groq rate limit handling
GROQ_RETRY_ATTEMPTS = 3
GROQ_RETRY_DELAY = 1  # seconds, increases exponentially

# Simple in-memory cache
CACHE = {}

# ======================================================
# UTILITIES
# ======================================================

def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json(text: str) -> dict:
    """Safely clean AI markdown and extract a JSON dictionary."""
    text = re.sub(r"```json\s?|\s?```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("❌ No valid JSON detected in AI output")
        return json.loads(match.group(0))


def generate_confidence(status: str) -> float:
    status = status.lower()
    ranges = {
        "normal": (0.85, 0.98),
        "low": (0.75, 0.90),
        "high": (0.75, 0.90),
        "critical": (0.65, 0.85)
    }
    low, high = ranges.get(status, (0.60, 0.80))
    return round(random.uniform(low, high), 2)


# ======================================================
# SMART RISK SCORING (USING ORGAN_SCORING MODULE)
# ======================================================

def calculate_risk_score(parameters: list) -> dict:
    """
    Calculate health score using the organ_scoring module.
    This provides accurate scoring based on reference ranges.
    """
    return calculate_health_score(parameters)


# ======================================================
# NORMALIZATION
# ======================================================

def normalize_result(data: dict, engine: str, proc_time: float, cache_hit: bool) -> dict:
    def safe(v, default=""):
        return v if v is not None else default

    # Extract raw parameters from AI
    raw_params = data.get("parameters", [])
    
    # ✅ VALIDATION: Check if we have actual parameters
    if not raw_params or len(raw_params) == 0:
        print("⚠️ WARNING: No parameters extracted from document. Returning minimal analysis.")
        return {
            "user_profile": {
                "name": safe(data.get("user_profile", {}).get("name"), "Patient"),
                "age": safe(data.get("user_profile", {}).get("age"), "N/A"),
                "gender": safe(data.get("user_profile", {}).get("gender"), "N/A"),
            },
            "parameters": [],
            "summary": "❌ No valid lab data available. Please upload a medical report with lab values.",
            "recommendations": ["Upload a valid medical report with biomarker values"],
            "doctor_perspective": "Unable to provide clinical assessment without lab data. Please provide a medical report containing laboratory test results.",
            "organ_scores": {"metabolic": 0, "cardiac": 0, "renal": 0, "hepatic": 0, "hematologic": 0},
            "health_plan": [],
            "disease_risks": [],
            "prevention_tips": [],
            "risk_metrics": {
                "health_score": 0,
                "overall_risk": "unknown",
                "critical_count": 0,
                "abnormal_count": 0,
                "normal_count": 0,
                "average_confidence": 0
            },
            "medical_disclaimer": (
                "⚕️ IMPORTANT: This analysis is generated by an AI system and is intended for "
                "informational and educational purposes only. It does NOT constitute medical advice, "
                "diagnosis, or treatment. AI systems can and do make errors. Always consult a qualified "
                "healthcare professional for medical decisions."
            ),
            "audit": {
                "analysis_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine": engine,
                "engine_version": ENGINE_VERSION,
                "processing_time_ms": int(proc_time * 1000),
                "cache_hit": cache_hit,
                "status": "no-parameters"
            }
        }
    
    # ✅ ENRICHMENT: Re-evaluate parameters using reference ranges
    enriched_params = enrich_parameters(raw_params)
    
    # Build parameter list with proper validation
    params = []
    for p in enriched_params:
        if not isinstance(p, dict):
            continue
        
        param_name = safe(p.get("name"), "Unknown Parameter")
        param_value = p.get("value")
        
        # Validate numeric value
        is_valid, numeric_val = validate_parameter_value(param_value)
        
        status = str(p.get("status", "normal")).lower().strip()
        
        params.append({
            "name": param_name,
            "value": f"{numeric_val}" if is_valid else safe(param_value, "N/A"),
            "unit": safe(p.get("unit")),
            "normalRange": safe(p.get("normalRange")),
            "status": status,
            "confidence": float(p.get("confidence", 0.92)),
            "explanation": safe(p.get("explanation"), f"Biomarker level is {status}."),
            "red_flag": status == "critical"
        })
    
    # ✅ VALIDATION: Only calculate scores if we have valid parameters
    if not params:
        print("⚠️ WARNING: Parameters could not be parsed into valid numeric values.")
        return {
            "user_profile": {
                "name": safe(data.get("user_profile", {}).get("name"), "Patient"),
                "age": safe(data.get("user_profile", {}).get("age"), "N/A"),
                "gender": safe(data.get("user_profile", {}).get("gender"), "N/A"),
            },
            "parameters": [],
            "summary": "❌ Could not parse lab values. Please ensure the document contains readable lab data.",
            "recommendations": ["Verify document quality and try again"],
            "doctor_perspective": "Unable to provide clinical assessment due to data parsing issues.",
            "organ_scores": {"metabolic": 0, "cardiac": 0, "renal": 0, "hepatic": 0, "hematologic": 0},
            "health_plan": [],
            "disease_risks": [],
            "prevention_tips": [],
            "risk_metrics": {
                "health_score": 0,
                "overall_risk": "unknown",
                "critical_count": 0,
                "abnormal_count": 0,
                "normal_count": 0,
                "average_confidence": 0
            },
            "medical_disclaimer": (
                "⚕️ IMPORTANT: This analysis is generated by an AI system and is intended for "
                "informational and educational purposes only."
            ),
            "audit": {
                "analysis_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine": engine,
                "engine_version": ENGINE_VERSION,
                "processing_time_ms": int(proc_time * 1000),
                "cache_hit": cache_hit,
                "status": "parse-error"
            }
        }
    
    # ✅ Calculate health score and risk metrics using enriched parameters
    risk_metrics = calculate_risk_score(params)

    # ✅ Calculate organ scores based on actual parameter data
    organ_scores = calculate_organ_scores(params)

    # Normalize health plan
    health_plan = []
    for day in data.get("health_plan", []):
        health_plan.append({
            "day": int(day.get("day", len(health_plan) + 1)),
            "focus": safe(day.get("focus"), "General wellness"),
            "diet": safe(day.get("diet"), "Balanced nutrition"),
            "exercise": safe(day.get("exercise"), "Light activity"),
            "precautions": safe(day.get("precautions"), "Stay hydrated"),
            "sleep": safe(day.get("sleep"), "7-8 hours"),
            "supplements": safe(day.get("supplements"), "As prescribed")
        })

    # Normalize disease risks
    disease_risks = []
    for risk in data.get("disease_risks", []):
        disease_risks.append({
            "disease": safe(risk.get("disease"), "Unknown"),
            "risk_level": safe(risk.get("risk_level"), "low"),
            "probability": max(0, min(100, int(risk.get("probability", 10)))),
            "explanation": safe(risk.get("explanation"), "")
        })

    return {
        "user_profile": {
            "name": safe(data.get("user_profile", {}).get("name"), "Patient"),
            "age": safe(data.get("user_profile", {}).get("age"), "N/A"),
            "gender": safe(data.get("user_profile", {}).get("gender"), "N/A"),
        },
        "parameters": params,
        "summary": safe(data.get("summary"), "Medical report analysis complete."),
        "recommendations": data.get("recommendations", []),
        "doctor_perspective": safe(
            data.get("doctor_perspective"),
            "Based on the available data, a comprehensive clinical evaluation is recommended. "
            "Some biomarkers may warrant further investigation to establish a clear diagnostic picture."
        ),
        "organ_scores": organ_scores,
        "health_plan": health_plan,
        "disease_risks": disease_risks,
        "prevention_tips": data.get("prevention_tips", [
            "Maintain regular health checkups",
            "Stay hydrated with adequate water intake",
            "Follow a balanced diet rich in nutrients"
        ]),
        "risk_metrics": risk_metrics,
        "medical_disclaimer": (
            "⚕️ IMPORTANT: This analysis is generated by an AI system and is intended for "
            "informational and educational purposes only. It does NOT constitute medical advice, "
            "diagnosis, or treatment. AI systems can and do make errors. Always consult a qualified "
            "healthcare professional for medical decisions. Never disregard professional medical "
            "advice or delay seeking it based on AI-generated content."
        ),
        "audit": {
            "analysis_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine": engine,
            "engine_version": ENGINE_VERSION,
            "processing_time_ms": int(proc_time * 1000),
            "cache_hit": cache_hit
        }
    }


# ======================================================
# ENHANCED AI PROMPT
# ======================================================

def build_prompt(text: str) -> str:
    return f"""You are MediQ, an expert, empathetic medical AI acting as a real consulting doctor. Analyze this medical document and return ONLY valid JSON.

RULES:
- Extract all biomarker values from the document.
- Assess each value against standard reference ranges.
- Be accurate - do NOT make up values not in the document.
- Use bullet points for all summaries and recommendations.
- Be specific and actionable.
- Return ONLY valid JSON, no markdown or explanation outside of it.

For the `doctor_perspective` field, adhere STRICTLY to these guidelines:
1. Speak exactly like a real, professional, and reassuring doctor in a consultation. Do NOT sound like an AI.
2. Clearly identify and list WHICH specific parameters are marked as "critical" or "abnormal" in their report.
3. Explain what these critical values actually mean for their body in simple, easy-to-understand terms.
4. Guide them with clear, actionable steps on how they can bring those specific values back to normal (including diet, lifestyle changes, precautions, or further tests).
5. Maintain a natural, warm, and professional tone throughout the explanation.

Return this exact JSON:
{{
  "user_profile": {{"name": "patient", "age": "age", "gender": "gender"}},
  "parameters": [{{"name": "biomarker", "value": "123", "unit": "unit", "normalRange": "range", "status": "normal", "explanation": "explanation"}}],
  "summary": "Finding 1. Finding 2. Main recommendations.",
  "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],
  "doctor_perspective": "A reassuring, doctor-like consultation clearly explaining the critical values, what they mean, and actionable steps (diet, lifestyle, etc.) to normalize them.",
  "organ_scores": {{"metabolic": 75, "cardiac": 70, "renal": 80, "hepatic": 75, "hematologic": 70}},
  "health_plan": [{{"day": 1, "focus": "focus", "diet": "meals", "avoid": "items", "exercise": "activity", "precautions": "notes", "sleep": "8 hours", "supplements": "none"}}],
  "disease_risks": [{{"disease": "condition", "risk_level": "low", "probability": 25, "explanation": "Why and symptoms"}}],
  "prevention_tips": ["Tip 1 with explanation", "Tip 2 with explanation"]
}}

MEDICAL DOCUMENT:
{text[:5000]}
"""


# ======================================================
# AI LAYERS
# ======================================================

def analyze_with_gemini(text: str) -> dict:
    if not gemini_client:
        raise RuntimeError("Gemini client not initialized (no API key).")
    prompt = build_prompt(text)
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return extract_json(response.text)


def analyze_with_groq(text: str) -> dict:
    """
    Analyze medical text using Groq (LLaMA 3.1) with retry logic for rate limiting.
    Implements exponential backoff for transient failures.
    """
    if not groq_client:
        raise RuntimeError("Groq client not initialized (no API key provided in .env)")
    
    prompt = build_prompt(text)
    
    # Retry logic with exponential backoff
    for attempt in range(GROQ_RETRY_ATTEMPTS):
        try:
            chat = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temp for deterministic medical analysis
                max_tokens=4096,
            )
            return extract_json(chat.choices[0].message.content)
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors
            if "rate_limit" in error_str or "429" in error_str:
                if attempt < GROQ_RETRY_ATTEMPTS - 1:
                    wait_time = GROQ_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"⏱️ Groq rate limited. Retrying in {wait_time}s (attempt {attempt + 1}/{GROQ_RETRY_ATTEMPTS})...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Groq rate limited after {GROQ_RETRY_ATTEMPTS} attempts: {e}")
            
            # Check for authentication errors
            elif "authentication" in error_str or "401" in error_str or "unauthorized" in error_str:
                raise RuntimeError(f"Groq authentication failed - invalid API key: {e}")
            
            # Check for model not found
            elif "model" in error_str or "not found" in error_str:
                raise RuntimeError(f"Groq model '{GROQ_MODEL}' not found or not available: {e}")
            
            # Other errors - no retry
            else:
                raise RuntimeError(f"Groq API error: {e}")
    
    raise RuntimeError("Groq analysis failed after all retry attempts")


# ======================================================
# ORCHESTRATOR
# ======================================================

def analyze_medical_text(text: str) -> dict:
    start_time = time.time()
    text_hash = get_text_hash(text)

    # Cache Check
    if text_hash in CACHE:
        cached = CACHE[text_hash].copy()
        cached["audit"] = cached.get("audit", {}).copy()
        cached["audit"]["cache_hit"] = True
        cached["audit"]["engine"] = "cache"
        return cached

    print("\n" + "="*60)
    print("🚀 ANALYSIS PIPELINE STARTED")
    print("="*60)
    
    # STEP 1: Direct biomarker parsing (before AI)
    print("\n[STEP 1/3] Direct biomarker extraction from PDF text...")
    direct_params = parse_biomarkers_from_text(text)
    
    # STEP 2: AI analysis
    print("\n[STEP 2/3] AI-based medical analysis...")
    engine_used = None
    raw = None
    
    # Try Gemini first
    if gemini_initialized:
        try:
            print(f"   Attempting Gemini ({GEMINI_MODEL})...")
            raw = analyze_with_gemini(text)
            engine_used = "gemini"
            print(f"   ✅ Gemini analysis successful")
        except Exception as e:
            print(f"   ❌ Gemini failed: {str(e)[:100]}")
            if groq_initialized:
                print(f"   Attempting fallback to Groq ({GROQ_MODEL})...")
    
    # Try Groq if Gemini didn't work
    if raw is None and groq_initialized:
        try:
            raw = analyze_with_groq(text)
            engine_used = "groq"
            print(f"   ✅ Groq analysis successful")
        except Exception as e:
            print(f"   ❌ Groq failed: {str(e)[:100]}")
    
    # Use error fallback if both failed
    if raw is None:
        print(f"   ⚠️ Both AI engines unavailable. Using error fallback.")
        if not gemini_initialized and not groq_initialized:
            print(f"      (Reason: No API keys configured. Add GEMINI_API_KEY or GROQ_API_KEY to .env)")
        raw = {
            "user_profile": {},
            "parameters": [],
            "summary": "AI processing unavailable. Using direct biomarker extraction only.",
            "doctor_perspective": "Unable to generate AI analysis at this time. Review the extracted parameters below.",
            "organ_scores": {},
            "health_plan": [],
            "disease_risks": [],
            "prevention_tips": [],
            "recommendations": []
        }
        engine_used = "error-fallback"

    # STEP 3: Merge direct parser results with AI results
    print("\n[STEP 3/3] Merging extraction results...")
    ai_params = raw.get("parameters", [])
    merged_params = validate_and_merge_parameters(ai_params, direct_params)
    
    # Update raw data with merged parameters
    raw["parameters"] = merged_params
    
    # Log summary
    print(f"\n📊 FINAL EXTRACTION SUMMARY:")
    print(f"   AI Engine Used: {engine_used.upper()}")
    print(f"   Direct parser found: {len(direct_params)} biomarkers")
    print(f"   AI analysis found: {len(ai_params)} biomarkers")
    print(f"   Merged result: {len(merged_params)} biomarkers")
    if len(merged_params) > 0:
        print(f"   Biomarkers: {', '.join([p.get('name', 'Unknown') for p in merged_params[:5]])}")
        if len(merged_params) > 5:
            print(f"   ... and {len(merged_params) - 5} more")

    normalized = normalize_result(
        raw,
        engine_used,
        time.time() - start_time,
        False
    )

    CACHE[text_hash] = normalized
    return normalized