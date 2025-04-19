#!/usr/bin/env python
import os
import re

def fix_imports(file_path):
    """
    Update import statements in Python files
    - Change 'from models.' to 'from pycil2.models.'
    - Change 'from utils.' to 'from pycil2.utils.'
    - Change 'from convs.' to 'from pycil2.convs.'
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex patterns to match import statements
    patterns = [
        (r'from\s+models\.', 'from pycil2.models.'),
        (r'from\s+utils\.', 'from pycil2.utils.'),
        (r'from\s+convs\.', 'from pycil2.convs.'),
        (r'import\s+models(?![a-zA-Z0-9_])', 'import pycil2.models'),
        (r'import\s+utils(?![a-zA-Z0-9_])', 'import pycil2.utils'),
        (r'import\s+convs(?![a-zA-Z0-9_])', 'import pycil2.convs'),
    ]
    
    modified_content = content
    for pattern, replacement in patterns:
        modified_content = re.sub(pattern, replacement, modified_content)
    
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        print(f"Updated imports in {file_path}")
        return True
    return False

def process_directory(directory):
    """Process all Python files in a directory and its subdirectories."""
    updated_files = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports(file_path):
                    updated_files += 1
    return updated_files

if __name__ == "__main__":
    pycil2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pycil2')
    updated = process_directory(pycil2_dir)
    print(f"Total files updated: {updated}")