#!/usr/bin/env python
import os
import shutil
import sys

def remove_redundant_files():
    """Remove files and directories that have been moved to the pycil2 package."""
    # Define root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directories to remove completely (they've been copied to pycil2/*)
    dirs_to_remove = ['convs', 'models', 'exps', 'resources']
    
    # Individual files that have been moved to pycil2/
    files_to_remove = ['main.py', 'trainer.py', '__init__.py']
    
    # Utility scripts no longer needed after restructuring
    utility_scripts = ['reorganize.py', 'fix_imports.py']
    
    # Remove directories
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
    
    # Remove utils directory but handle it separately since we'll keep some files
    utils_dir = os.path.join(root_dir, 'utils')
    if os.path.exists(utils_dir) and os.path.isdir(utils_dir):
        print(f"Removing directory: {utils_dir}")
        shutil.rmtree(utils_dir)
    
    # Remove individual files
    for file_name in files_to_remove + utility_scripts:
        file_path = os.path.join(root_dir, file_name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"Removing file: {file_path}")
            os.remove(file_path)
    
    print("\nCleaning completed successfully!")

if __name__ == "__main__":
    # Prompt for confirmation to avoid accidental deletion
    response = input("This will remove all redundant files and directories that have been moved to the pycil2 package.\nAre you sure you want to continue? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        remove_redundant_files()
    else:
        print("Cleanup cancelled.")