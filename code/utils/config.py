import os
import yaml
from typing import Dict, Any
import re
from pathlib import Path


def load_config(config_path: str) -> Dict[Any, Any]:
    """
    Load YAML configuration with support for inheritance and variable interpolation.
    This function reads a YAML configuration file and processes it to:
    1. Handle inheritance through the 'includes' directive, merging parent configs
    2. Resolve variable interpolations in the format ${variable.path}
    Args:
        config_path (str): Path to the YAML configuration file to load
    Returns:
        Dict[Any, Any]: The processed configuration dictionary with all includes
                        merged and variables resolved
    Notes:
        - Included configurations are merged with the current one, with the current
          configuration taking precedence over included ones
        - The 'includes' key is removed from the final configuration
        - Paths in 'includes' are resolved relative to the current config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'includes' in config:
        base_dir = os.path.dirname(config_path)
        for include_path in config['includes']:
            full_path = os.path.normpath(os.path.join(base_dir, include_path))
            include_config = load_config(full_path)
            config = deep_merge(include_config, config)
        del config['includes']
    
    config = resolve_variables(config)
    
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_variables(config: Dict) -> Dict:
    config_str = yaml.dump(config)
    
    # Pattern to find ${variable.path} references
    pattern = r'\${([\w.]+)}'
    
    def replace_var(match):
        var_path = match.group(1)
        keys = var_path.split('.')
        value = config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return match.group(0)  # Keep original if not found
        return str(value)
    
    while re.search(pattern, config_str):
        config_str = re.sub(pattern, replace_var, config_str)
    
    return yaml.safe_load(config_str)
