import os
import json
import shutil

def find_key(data, key):
    """Recursively yield values associated with a specified key in a nested structure."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                yield v.split('_')[0] if isinstance(v, str) else v
            elif isinstance(v, (dict, list)):
                yield from find_key(v, key)
    elif isinstance(data, list):
        for item in data:
            yield from find_key(item, key)

def process_files(directory, key):
    """Process JSON files in a directory, categorizing them by key values."""
    file_paths, types = [], []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        values = list(find_key(data, key))
                        if len(values) == 1:
                            file_paths.append(path)
                            types.append(values[0])
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing file {path}: {e}")

    # Create folders and copy files with unique names
    output_directory = "../data/type_wise_cdm_samples_json"
    for file_path, file_type in zip(file_paths, types):
        folder_path = os.path.join(output_directory, file_type)
        os.makedirs(folder_path, exist_ok=True)

        base_name = os.path.basename(file_path)
        target_path = os.path.join(folder_path, base_name)
        count = 1

        while os.path.exists(target_path):
            name, ext = os.path.splitext(base_name)
            target_path = os.path.join(folder_path, f"{name}_{count}{ext}")
            count += 1

        shutil.copy(file_path, target_path)

if __name__ == "__main__":
    process_files("../data/cdm_samples_json", "productQualifier")