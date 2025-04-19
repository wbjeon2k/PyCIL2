#!/usr/bin/env python
import os
import shutil
import sys

def move_files(src_dir, dest_dir, extensions=None):
    """Move files from src_dir to dest_dir, preserving only certain extensions if specified."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        # Skip if it's a directory we shouldn't copy
        if os.path.isdir(src_path) and item not in ['__pycache__', 'pycil2', 'build', '.git', '.vscode', '.devcontainer']:
            # For rl_utils directory special case, recurse but don't create another rl_utils
            if item == 'rl_utils':
                move_files(src_path, os.path.join(dest_dir), extensions)
            else:
                move_files(src_path, os.path.join(dest_dir, item), extensions)
        elif os.path.isfile(src_path):
            # If extensions are provided, only copy files with those extensions
            if extensions is None or any(item.endswith(ext) for ext in extensions):
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")

def main():
    # Define base directories
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup directory mappings
    dirs_to_move = {
        'models': 'pycil2/models',
        'utils': 'pycil2/utils',
        'convs': 'pycil2/convs',
        'exps': 'pycil2/exps',
        'resources': 'pycil2/resources',
    }
    
    # Move the files
    for src, dest in dirs_to_move.items():
        src_dir = os.path.join(root_dir, src)
        dest_dir = os.path.join(root_dir, dest)
        print(f"Moving files from {src_dir} to {dest_dir}")
        
        # Only move Python files and JSON files as appropriate
        extensions = ['.py'] if src != 'exps' else ['.json']
        move_files(src_dir, dest_dir, extensions)
    
    print("Done reorganizing files")

if __name__ == "__main__":
    main()