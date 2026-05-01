import os
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.cwd() / '.env', override=True)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
print('🔍 Groq API Diagnostic Test')
print('=' * 60)
print(f'API Key: {GROQ_API_KEY[:30]}...' if GROQ_API_KEY else 'NO API KEY')

from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

# Simple test
prompt = 'Respond with valid JSON only: {"test": "working"}'

try:
    print('\n📤 Sending test request to Groq...')
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=200
    )
    print('✅ Groq API Works!')
    print(f'Response: {response.choices[0].message.content[:100]}')
except Exception as e:
    print(f'❌ Error: {type(e).__name__}')
    print(f'Message: {str(e)}')
