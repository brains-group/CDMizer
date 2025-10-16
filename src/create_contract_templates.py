import os
import json
from datetime import datetime
from typing import Any, Dict, Set

def shorten_cdm(data: Any) -> Any:
    """
    Recursively remove unnecessary keys from a given data structure.

    Args:
        data: The input data (dictionary, list, or primitive type).

    Returns:
        The data with specific keys removed.
    """
    exclude = ['globalKey', 'scheme', 'globalReference', 'partyId', 'assignedIdentifier']
    if isinstance(data, dict):
        return {k: v for k, v in ((k, shorten_cdm(v)) for k, v in data.items() if k not in exclude) if v}
    elif isinstance(data, list):
        return [item for item in (shorten_cdm(item) for item in data) if item]
    else:
        return data

def clean_empty(data):
    """
    Recursively removes empty objects, empty lists, or lists containing only empty objects from a JSON structure.
    Additionally, replaces objects with only a 'description' key with an empty string.
    """
    if isinstance(data, dict):
        # Recursively clean the dictionary
        cleaned_dict = {}
        for k, v in data.items():
            # Replace objects with only a 'description' key with an empty string
            if isinstance(v, dict) and set(v.keys()) == {"description"}:
                cleaned_dict[k] = ""
            else:
                cleaned_dict[k] = clean_empty(v)
        # Remove keys with empty values
        return {k: v for k, v in cleaned_dict.items() if v != {} and v != []}
    elif isinstance(data, list):
        # Recursively clean each item in the list
        cleaned_list = [clean_empty(item) for item in data]
        # Remove empty lists or empty objects from the list
        return [item for item in cleaned_list if item != {} and item != []]
    else:
        # Return the data as is if it's neither a list nor a dictionary
        return data

def load_json_schema(file_path: str) -> Dict:
    """
    Load a JSON schema from a file.

    Args:
        file_path: Path to the JSON schema file.

    Returns:
        The loaded JSON schema as a dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_dummy_data(data_type: str) -> Any:
    """
    Generate dummy data based on the specified data type.

    Args:
        data_type: The type of data to generate.

    Returns:
        Dummy data of the specified type.
    """
    if data_type == "string":
        return ""
    elif data_type == "array":
        return []
    elif data_type == "object":
        return {}
    elif "date" in data_type.lower():
        return datetime.now().strftime("%Y-%m-%d")
    elif "time" in data_type.lower():
        return datetime.now().strftime("%H:%M:%S")
    else:
        return None

def load_example_structures(example_folder: str) -> Set[str]:
    """
    Load example JSON files and extract their key structures.

    Args:
        example_folder: Path to the folder containing example JSON files.

    Returns:
        A set of flattened keys from all example JSON files.
    """
    structures = set()
    for file_name in os.listdir(example_folder):
        file_path = os.path.join(example_folder, file_name)
        if file_name.endswith(".json") and os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                example = json.load(file)
                flattened_keys = flatten_keys(example)
                structures.update(flattened_keys)
    return structures

def flatten_keys(data: Any, prefix: str = "") -> Set[str]:
    """
    Flatten nested dictionary keys into dot-separated strings.

    Args:
        data: The input data (dictionary or list).
        prefix: The current key prefix.

    Returns:
        A set of flattened keys.
    """
    keys = set()
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            keys.update(flatten_keys(value, full_key))
    elif isinstance(data, list):
        for item in data:
            keys.update(flatten_keys(item, prefix))
    return keys

def create_parse_tree(file_path: str, base_folder: str, example_keys: Set[str], current_prefix="trade") -> Dict:
    """
    Recursively create a parse tree from JSON schemas.

    Args:
        file_path: Path to the schema file.
        base_folder: Base folder containing the schemas.
        example_keys: Set of example keys for pruning.
        current_prefix: Current key prefix.

    Returns:
        A dictionary representing the parse tree.
    """

    schema = load_json_schema(file_path)
    description = schema.get("description", "")
    parse_tree = {"description": description} if description else {}

    for prop, details in schema.get("properties", {}).items():
        prop_prefix = f"{current_prefix}.{prop}" if current_prefix else prop
     
        if prop_prefix not in example_keys:
            continue

        if "$ref" in details:
            ref_path = os.path.join(base_folder, details["$ref"])
            # modified_ref_path = ref_path.replace('ReferenceWith', 'FieldWith')
            # if os.path.exists(modified_ref_path):
            #     ref_path = modified_ref_path
            parse_tree[prop] = create_parse_tree(ref_path, base_folder, example_keys, prop_prefix)
        elif details.get("type") == "array":
            parse_tree[prop] = []
            if "items" in details and "$ref" in details["items"]:
                ref_path = os.path.join(base_folder, details["items"]["$ref"])
                # modified_ref_path = ref_path.replace('ReferenceWith', 'FieldWith')
                # if os.path.exists(modified_ref_path):
                #     ref_path = modified_ref_path
                parse_tree[prop].append(create_parse_tree(ref_path, base_folder, example_keys, prop_prefix))
            else:
                parse_tree[prop].append(generate_dummy_data(details["items"].get("type", "string")))
        elif details.get("type"):
            parse_tree[prop] = generate_dummy_data(details["type"])

    return parse_tree

def build_tree_for_directory(schema_folder: str, example_base_folder: str, output_folder: str):
    """
    Generate parse trees for all subdirectories in the example base folder.

    Args:
        schema_folder: Path to the schema folder.
        example_base_folder: Path to the base folder containing example subdirectories.
        output_folder: Path to save the parse tree files.
    """
    for subdir in os.listdir(example_base_folder):
        example_folder = os.path.join(example_base_folder, subdir)
        if os.path.isdir(example_folder):
            start_schema = "cdm-event-common-Trade.schema.json"
            start_file_path = os.path.join(schema_folder, start_schema)
            example_keys = load_example_structures(example_folder)

            parse_tree = {"trade": create_parse_tree(start_file_path, schema_folder, example_keys)}
            parse_tree = clean_empty(parse_tree)

            output_file = os.path.join(output_folder, f"template_{subdir}.json")
            save_parse_tree_to_file(parse_tree, output_file)
            print(f"Parse tree for '{subdir}' has been saved to {output_file}.")

def save_parse_tree_to_file(parse_tree: Dict, output_file: str):
    """
    Save the parse tree to a JSON file.

    Args:
        parse_tree: The parse tree dictionary to save.
        output_file: Path to the output JSON file.
    """
    with open(output_file, 'w') as file:
        json.dump(parse_tree, file, indent=4)

schema_folder = "cdm_schema/cdm_schema_json"
example_base_folder = "../data/type_wise_cdm_samples_json"
output_folder = "cdm_schema/type_wise_templates"

os.makedirs(output_folder, exist_ok=True)

build_tree_for_directory(schema_folder, example_base_folder, output_folder)
