import pandas as pd
import json
import os
from llm_handler import *
import re


def extract_unique_keys_from_json(data):
    keys = set()
    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(key)
            keys.update(extract_unique_keys_from_json(value))
    elif isinstance(data, list):
        for item in data:
            keys.update(extract_unique_keys_from_json(item))
    return keys


def get_valid_unmatched_keys(cdm_schema_path, cdm_repo_path):
    
    cdm_schema_df = pd.read_csv(cdm_schema_path+'cdm_schema.csv')
    cdm_keys = cdm_schema_df['Attribute Name'].unique().tolist()

    unmatched_keys = set() 
    
    for root, dirs, files in os.walk(cdm_repo_path):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as json_file:
                    cdm_repr = json.load(json_file)
                    json_keys = extract_unique_keys_from_json(cdm_repr)
                    incorrect_keys = json_keys.difference(set(cdm_keys))
                    unmatched_keys.update(incorrect_keys)
    
    return unmatched_keys


def compare_keys(cdm_repr, cdm_schema_df, valid_unmatched_keys):

    cdm_keys = {str(key).lower() for key in cdm_schema_df['Attribute Name'].unique()}
    cdm_keys.update({str(key).lower() for key in valid_unmatched_keys})
    
    json_keys = {str(key).lower() for key in extract_unique_keys_from_json(cdm_repr)}

    correct_keys = json_keys.intersection(set(cdm_keys))
    correct_percentage = (len(correct_keys)/len(json_keys))*100 if json_keys else 0
    incorrect_keys = json_keys.difference(set(cdm_keys))
    
    return {
        "total_json_keys": len(json_keys),
        "total_correct_keys": len(correct_keys),
        "correct_percentage": correct_percentage,
        "incorrect_keys": list(incorrect_keys)
    }


def load_schemas(cdm_schema_json_path):
   
    schemas = {}
    
    for schema_file in os.listdir(cdm_schema_json_path):
        try:
            if schema_file.endswith('.json'):
                typename = schema_file.split('-')[-1].split('.')[0].lower()
                file_path = os.path.join(cdm_schema_json_path, schema_file)
                
                with open(file_path, 'r') as file:
                    schema_data = json.load(file)
                    schemas[typename] = schema_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {schema_file}, error: {e}")
        except Exception as e:
            print(f"Error loading file: {schema_file}, error: {e}")
    
    return schemas



def validate_object_against_schema(obj, schema):

    schema_keys = set(schema.get("properties", []))
    schema_keys.update(['meta', 'value'])
    obj_keys = set(obj.keys())

    matched_keys_cnt = len(obj_keys.intersection(schema_keys))
    total_keys = len(obj_keys)
    
    return matched_keys_cnt, total_keys
    

def compare_schema(cdm_repr, schemas, cdm_keys):
    
    matched_keys = 0
    total_keys = 0
   
    for key, value in cdm_repr.items():
        typename = key.lower()
        if isinstance(value, dict):
            matched_curr = 0
            total_curr = 0
            if typename in schemas:
                schema = schemas[typename]
                matched_curr, total_curr = validate_object_against_schema(value, schema)
            
            matched_next, total_next = compare_schema(value, schemas, cdm_keys)
            
            matched_keys += matched_curr + matched_next
            total_keys += total_curr + total_next
                
        elif isinstance(value, list):
            for obj in value:
                if isinstance(obj, dict):
                    matched_curr, total_curr = compare_schema(obj, schemas, cdm_keys)
                    matched_keys += matched_curr
                    total_keys += total_curr
        else:            
            if typename in cdm_keys:
                matched_keys += 1
            total_keys += 1
            
    return matched_keys, total_keys



def get_coverage_score(coverage):

    pattern = r'"(?P<key>\w+)":\s*\[(?P<list>(?:\([^)]*\)|[^]])*?)\]'
    matches = re.finditer(pattern, coverage, re.DOTALL)

    coverage_dict = {}
    for match in matches:
        key = match.group("key")
        raw_list = match.group("list").strip()
        raw_elements = [item.strip() for item in raw_list.split('\n')] if raw_list else []
        coverage_dict[key] = raw_elements
    

    try:
        tot_matched = len(coverage_dict['matched_info'])
    except:
        tot_matched = 0
        print("Total matched = 0!!!")
    
    try:
        tot_uncaptured = len(coverage_dict['contract_has_but_cdm_doesnt'])
    except:
        tot_uncaptured = 0
        print("Total uncaptured = 0!!!")

    
    try:
        extraneous = coverage_dict['cdm_has_but_contract_doesnt']
    except:
        extraneous = []
        print("Total extraneous = 0!!!")

    exclude = {"meta", "global", "external", "scheme", "identifier", "Identifier"}

    filtered_extraneous = [
        str(element) for element in extraneous
        if not any(excl in str(element) for excl in exclude)
    ]

    tot_extraneous = len(filtered_extraneous)

    if (tot_matched+0.3*tot_uncaptured+0.1*tot_extraneous) == 0:
        print("Evaluation failed for this!!")
        return {
            'score': 'undefined',
            'coverage_dict': coverage_dict
        }

    score = (tot_matched*100)/(tot_matched+0.3*tot_uncaptured+0.1*tot_extraneous)

    coverage_dict['cdm_has_but_contract_doesnt'] = filtered_extraneous

    ret = {
        'score': score,
        'coverage_dict': coverage_dict
    }

    return ret
    

def get_coverage(llm, contract_desc, cdm_repr):

    context = """I will give you a contract description in natural language and an ISDA CDM (Common Domain Model) representation for the same contract. Your task is to identify if the CDM representation fully captures all the information from the contract description. Provide an output having the following three lists:

    1. **matched_info**: a list of information from the contract description that is correctly captured in the CDM representation.
    2. **contract_has_but_cdm_doesnt**: a list of information from the contract description that is not captured in the CDM representation.
    3. **cdm_has_but_contract_doesnt**: a list of 'key-value' pairs in the CDM representation that are not present in the contract description.

    The output (ONLY A DICT) MUST FOLLOW EXACTLY the following format:

    {
        "matched_info": [],
        "contract_has_but_cdm_doesnt": [],
        "cdm_has_but_contract_doesnt": []
    }

    ### Instructions:

    To achieve the output, PRECISELY follow these steps:

    # Steps

    1. **Extract Key Information from the Contract Description**:
    - Create an empty list to store the key information.
    - For each sentence in the contract description:
        - Extract the **important parts** of the sentence as key-value pairs. Do NOT include sentences, phrases, or just headings that do not contain any key-information. ONLY include those which can be represented in key value pairs.
          For example, given the sentence:
            “This contract represents a Credit Default Swap on a basket of reference entities, with Party 1 as the protection seller and Party 2 as the protection buyer.
            You should extract the following important parts:
            - `'contract: credit default swap'`
            - `'seller: Party 1'`
            - `'buyer: Party 2'`
        - If the sentence is not very informative, ignore it.
        - Any NOUNS, numbers, IDs MUST be included.
        - Append each important part to the list.

    2. **Create the Comparison Lists**:
    - After processing all sentences, the list will contain all key information from the contract description.
    - Now, initialize three empty lists: **matched_info**, **contract_has_but_cdm_doesnt**, and **cdm_has_but_contract_doesnt**.

    3. **Compare Contract Information with the CDM Representation**:
    - For each item in the list of important parts from the contract description:
        - Check if this item is captured in the CDM representation. 
        This does not have to be exact match. We are looking for semantic match. For example, 
        A DATE CAN BE IN DIFFERENT FORMATS IN THE CDM AND CONTRACT DESCRIPTION, OR, SAME INFORMATION
        CAN BE WORDED DIFFERENTLY IN THE TWO PLACES.
        - If yes, append it to the **matched_info** list.
        - If not, append it to **contract_has_but_cdm_doesnt**.
    - If any value in the CDM representation is unmatched with any information from the contract description,
        append it to "cdm_has_but_contract_doesnt" list.
        - For this list, please ignore any ID numbers, global/meta identifiers, etc

    # Expected Output Format

    Provide the following in Python list format:

    - **matched_info**: List of items from the contract description that is correctly captured in the CDM representation.
    - **contract_has_but_cdm_doesnt**: List of items from the contract description that are not captured in the CDM.
    - **cdm_has_but_contract_doesnt**: List of key-value pairs from the CDM that are not found in the contract description.

    Ensure all lists are returned in a structured Python format for easy interpretation.
    Remember, I DO NOT want any python code. I want YOU to do all the computations and finally 
    give just a Dictionary with the 3 lists  as output.
    
    Jut a reminder, the output (ONLY A DICT) MUST FOLLOW EXACTLY the following format:

    {
        "matched_info": [],
        "contract_has_but_cdm_doesnt": [],
        "cdm_has_but_contract_doesnt": []
    }

    """

    prompt = context + "\n\n ### Contract Description : \n" + contract_desc + "\n\n ### CDM Representation : \n" + str(cdm_repr)

    coverage = llm.generate(prompt)

    return coverage




def evaluate_cdm(cdm_json, cdm_schema_path, cdm_repo_path):

    valid_unmatched_keys = get_valid_unmatched_keys(cdm_schema_path, cdm_repo_path)
        
    cdm_schema_csv = pd.read_csv(cdm_schema_path+'cdm_schema.csv')
    
    key_correctness = compare_keys(cdm_json, cdm_schema_csv, valid_unmatched_keys)

    cdm_schemas = load_schemas(cdm_schema_path+'cdm_schema_json/')
    
    cdm_keys = set(cdm_schema_csv['Attribute Name'].unique())
    cdm_keys.update(valid_unmatched_keys)
    cdm_keys_lower = {str(key).lower() for key in cdm_keys}
    
    matched_keys, total_keys = compare_schema(cdm_json, cdm_schemas, cdm_keys_lower)
    schema_adherence = matched_keys*100/total_keys

    return key_correctness, schema_adherence
    
