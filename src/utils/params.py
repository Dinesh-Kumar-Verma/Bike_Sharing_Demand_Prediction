import yaml
from pathlib import Path

# Define the path to the params.yaml file relative to this script
# This assumes this file is in src/utils and params.yaml is in the root
PARAMS_FILE = Path(__file__).resolve().parent.parent.parent / "params.yaml"

def load_params():
    """Loads parameters from the params.yaml file."""
    try:
        with open(PARAMS_FILE, 'r') as f:
            params = yaml.safe_load(f)
        if params is None:
            return {}
        return params
    except FileNotFoundError:
        print(f"Error: '{PARAMS_FILE}' not found. Please ensure the file exists.")
        return {}
    except Exception as e:
        print(f"Error loading or parsing '{PARAMS_FILE}': {e}")
        return {}

