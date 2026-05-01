import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.cwd() / '.env', override=True)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

# Simplified medical prompt (testing)
prompt = """You are MediQ, a medical analysis AI. Return ONLY valid JSON.

Medical Document:
Patient John, Age 45, Male
Test Results:
- Glucose: 145 mg/dL
- Cholesterol: 220 mg/dL  
- Blood Pressure: 135/85 mmHg

Return this JSON:
{
  "user_profile": {
    "name": "John",
    "age": "45",
    "gender": "Male"
  },
  "parameters": [
    {
      "name": "Glucose",
      "value": "145",
      "unit": "mg/dL",
      "status": "high"
    }
  ],
  "summary": "Test analysis complete",
  "recommendations": ["Stay hydrated"],
  "doctor_perspective": "Medical findings show elevated glucose",
  "organ_scores": {"metabolic": 65},
  "health_plan": [],
  "disease_risks": [],
  "prevention_tips": []
}"""

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
    print('📤 Testing medical analysis prompt...')
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=1000
    )
    
    raw_response = response.choices[0].message.content
    print('\n✅ Got response from Groq')
    print(f'Raw response length: {len(raw_response)} chars')
    print(f'First 200 chars: {raw_response[:200]}')
    
    # Try to extract JSON
    try:
        data = extract_json(raw_response)
        print('\n✅ JSON extraction successful!')
        print(f'Keys: {list(data.keys())}')
        if 'parameters' in data:
            print(f'Parameters found: {len(data["parameters"])}')
    except Exception as e:
        print(f'\n❌ JSON extraction failed: {str(e)}')
        print(f'Raw response:\n{raw_response}')
        
except Exception as e:
    print(f'❌ API Error: {type(e).__name__}: {str(e)}')
