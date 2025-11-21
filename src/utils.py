import yaml
import os

def load_config(config_path='config.yaml'):
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def numerize_value(value):
    """
    A placeholder for a utility function (e.g., to format large numbers for KPIs).
    
    Example: 1250000 -> 1.25M
    You can install 'numerize' package and replace the logic here:
    from numerize import numerize
    return numerize.numerize(value)
    """
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.0f}"
