import os
import yaml
def get_config_Full_file():
    module_dir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.join(os.path.dirname(module_dir), 'config', 'config.yaml'))
    
    # Load and return the config data
    with open(config_path, 'r') as file:
        config_file_data = yaml.safe_load(file)
    
    return config_file_data
