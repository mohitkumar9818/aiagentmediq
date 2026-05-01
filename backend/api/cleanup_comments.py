import re

files_to_clean = [
    'random_forest_model.py',
    'svm_model.py', 
    'xgboost_model.py',
    'ml_models_integration.py'
]

def remove_comments(content):
    lines = content.split('\n')
    result = []
    in_docstring = False
    
    for line in lines:
        stripped = line.lstrip()
        
        if '"""' in line or "'''" in line:
            in_docstring = not in_docstring
            result.append(line)
            continue
        
        if in_docstring:
            result.append(line)
            continue
        
        if stripped.startswith('#'):
            continue
        
        if '#' in line and not in_docstring:
            code_part = line[:line.index('#')]
            if code_part.strip():
                result.append(code_part.rstrip())
        else:
            result.append(line)
    
    return '\n'.join(result)

for filename in files_to_clean:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = remove_comments(content)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f'✅ {filename} - Comments removed')
    except Exception as e:
        print(f'❌ {filename} - Error: {str(e)[:100]}')

print('\n✅ All comments removed!')
