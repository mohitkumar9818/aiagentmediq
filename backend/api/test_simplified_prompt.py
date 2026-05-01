import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.cwd() / '.env', override=True)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

# Simulate the simplified build_prompt
medical_doc = """
PATIENT REPORT
Name: John Doe
Age: 45
Gender: Male

LABORATORY RESULTS:
- Glucose: 145 mg/dL (Normal: 70-100)
- Total Cholesterol: 220 mg/dL (Normal: <200)
- HDL Cholesterol: 35 mg/dL (Normal: >40)
- LDL Cholesterol: 160 mg/dL (Normal: <100)
- Triglycerides: 200 mg/dL (Normal: <150)
- Blood Pressure: 135/85 mmHg (Normal: <120/80)
- Hemoglobin A1C: 6.8% (Normal: <5.7%)
- Creatinine: 1.0 mg/dL (Normal: 0.7-1.3)
- Bilirubin: 0.9 mg/dL (Normal: 0.1-1.2)
"""

prompt = f"""You are MediQ, a medical analysis AI. Analyze this medical document and return ONLY valid JSON.

RULES:
- Extract all biomarker values from the document
- Assess each value against standard reference ranges
- Be accurate - do NOT make up values not in the document
- Use bullet points (•) for all summaries and recommendations
- Be specific and actionable
- Return ONLY valid JSON, no markdown or explanation

JSON STRUCTURE:
{{
  "user_profile": {{"name": "patient name if found", "age": "age if found", "gender": "gender if found"}},
  "parameters": [
    {{"name": "biomarker", "value": "123", "unit": "unit", "normalRange": "range", "status": "normal/low/high/critical", "explanation": "brief explanation"}}
  ],
  "summary": "• Finding 1\\n• Finding 2\\n• Recommendation 1\\n• Recommendation 2",
  "recommendations": ["• Rec 1: explanation", "• Rec 2: explanation", "• Rec 3: explanation"],
  "doctor_perspective": "• What I found\\n• Why it matters\\n• What to focus on\\n• Next steps",
  "organ_scores": {{"metabolic": 75, "cardiac": 70, "renal": 80, "hepatic": 75, "hematologic": 70}},
  "health_plan": [
    {{"day": 1, "focus": "focus area", "diet": "• Breakfast: item\\n• Lunch: item", "avoid": "• avoid item 1", "exercise": "• Type: walking (30 min)", "precautions": "• precaution 1", "sleep": "• 8 hours", "supplements": "• Vitamin D: 1000 IU"}}
  ],
  "disease_risks": [
    {{"disease": "condition", "risk_level": "moderate", "probability": 35, "explanation": "• Why this risk\\n• What to watch"}}
  ],
  "prevention_tips": ["• Tip 1: how and why", "• Tip 2: how and why"]
}}

MEDICAL DOCUMENT:
{medical_doc[:8000]}
"""

def extract_json(text):
    text = re.sub(r"```json\s?|\s?```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON found")
        return json.loads(match.group(0))

try:
    print('🔬 Testing simplified medical prompt...')
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    
    raw_response = response.choices[0].message.content
    print('✅ Got response from Groq')
    print(f'Response length: {len(raw_response)} chars\n')
    
    # Try to extract JSON
    try:
        data = extract_json(raw_response)
        print('✅ JSON extraction SUCCESSFUL!')
        print(f'\n📊 Analysis Results:')
        print(f'  - User: {data.get("user_profile", {})}')
        print(f'  - Parameters found: {len(data.get("parameters", []))}')
        print(f'  - Health plan days: {len(data.get("health_plan", []))}')
        print(f'  - Disease risks: {len(data.get("disease_risks", []))}')
        print(f'  - Summary (first 100 chars): {str(data.get("summary", ""))[:100]}')
    except Exception as e:
        print(f'❌ JSON extraction failed: {str(e)}')
        print(f'\nRaw response:\n{raw_response[:500]}...')
        
except Exception as e:
    print(f'❌ API Error: {type(e).__name__}: {str(e)}')
