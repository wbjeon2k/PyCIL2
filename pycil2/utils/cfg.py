'''
Utility to manage config values
'''

import os
import json
import glob
from typing import List, Dict, Any, Optional


def list_all_exps() -> List[str]:
    """
    Lists all available experiment configurations in the exps directory.
    
    Returns:
        List[str]: A list of available experiment/model names (without .json extension)
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Build the path to the exps directory
    exps_dir = os.path.join(current_dir, 'exps')
    
    # Get all JSON files in the exps directory
    json_files = glob.glob(os.path.join(exps_dir, '*.json'))
    
    # Extract just the filenames without extensions and sort them
    model_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in json_files])
    
    return model_names


def print_available_models() -> None:
    """
    Prints a formatted list of all available models/experiment configurations.
    """
    models = list_all_exps()
    
    print("Available Models in PyCIL2:")
    print("-" * 40)
    
    # Print in columns if there are many models
    num_cols = 3
    col_width = 20
    
    for i in range(0, len(models), num_cols):
        row = models[i:i+num_cols]
        print("".join(model.ljust(col_width) for model in row))
    
    print("-" * 40)
    print(f"Total: {len(models)} models available")


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Retrieves the configuration for a specified model.
    
    Args:
        model_name (str): The name of the model (without .json extension)
        
    Returns:
        Dict[str, Any]: The model configuration as a dictionary
        
    Raises:
        FileNotFoundError: If the model configuration file doesn't exist
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(current_dir, 'exps', f'{model_name}.json')
    
    if not os.path.exists(config_path):
        available = list_all_exps()
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Available models: {', '.join(available)}"
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def search_models(keyword: str) -> List[str]:
    """
    Searches for models whose names contain the given keyword.
    
    Args:
        keyword (str): The keyword to search for in model names
        
    Returns:
        List[str]: List of model names containing the keyword
    """
    all_models = list_all_exps()
    return [model for model in all_models if keyword.lower() in model.lower()]


if __name__ == "__main__":
    # Example usage
    print_available_models()