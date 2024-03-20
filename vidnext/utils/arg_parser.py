
import yaml

def parse_yaml_file(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print("Error parsing YAML:", e)