import re

with open(r'e:\桌面\schooles\机器学习\C\C_model_reproduction\main.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix literal backslashes
code = code.replace('\\\\n', '\\n')

with open(r'e:\桌面\schooles\机器学习\C\C_model_reproduction\main.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Fixed backslashes')
