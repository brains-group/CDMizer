import sys
import os
import json
import re

src_directory = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(src_directory)

from llm_handler import *




def load_test_contracts(test_data_dir):
    text_dir = os.path.join(test_data_dir, "text")

    test_contracts = {}

    for sub_folder in os.listdir(text_dir):
        text_sub_folder = os.path.join(text_dir, sub_folder)
        
        if os.path.isdir(text_sub_folder):
            contracts = []
            for text_file in os.listdir(text_sub_folder):
                if text_file.endswith(".txt"):
                    text_path = os.path.join(text_sub_folder, text_file)
                    
                    if os.path.exists(text_path):
                       
                        with open(text_path, 'r') as tf:
                            text_content = tf.read()
                        
                        contracts.append({
                            "id": text_file,
                            "description": text_content
                        })
            
            if contracts:
                test_contracts[sub_folder] = contracts

    return test_contracts


def clean_cdm(cdm):
    """
    Recursively removes empty objects, empty lists, or lists containing only empty objects from a JSON structure.
    Additionally, removes 'description' keys from dictionaries and replaces objects with only a 'description' key with an empty string.
    """
    if isinstance(cdm, dict):
        cleaned_dict = {}
        for k, v in cdm.items():
            if k == "description":
                continue
            if isinstance(v, dict) and set(v.keys()) == {"description"}:
                cleaned_dict[k] = ""
            else:
                cleaned_dict[k] = clean_cdm(v)
        return {k: v for k, v in cleaned_dict.items() if v != {} and v != []}
    elif isinstance(cdm, list):
        cleaned_list = [clean_cdm(item) for item in cdm]
        return [item for item in cleaned_list if item != {} and item != []]
    else:
        return cdm

def extract_json(string):
    match = re.search(r'(\[.*\]|\{.*\})', string, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            corrected_json_string = re.sub(r"(?<!\\)'", '"', json_string)
            corrected_json_string = corrected_json_string.replace("True", "true").replace("False", "false").replace("None", "null")
            try:
                return json.loads(corrected_json_string)
            except json.JSONDecodeError:
                return None
    return None
    

def is_leaf(node):
    """Check if a node is a leaf (no nested dicts or lists)."""
    if isinstance(node, dict):
        return all(not isinstance(v, (dict, list)) for v in node.values())
    elif isinstance(node, list):
        return all(not isinstance(item, (dict, list)) for item in node)
    return True

def max_depth(node):
    """Calculate the maximum depth of a node."""
    if is_leaf(node):
        return 0
    if isinstance(node, dict):
        return 1 + max(max_depth(v) for v in node.values() if isinstance(v, (dict, list)))
    elif isinstance(node, list):
        return 1 + max(max_depth(item) for item in node if isinstance(item, (dict, list)))
    return 0

def create_prompt(llm_handler, contract_description, obj_definition, path, obj, use_rag=False):
    system_prompt = """
        You are a highly accurate assistant specializing in processing derivatives contract data and populating JSON templates. Your primary goal is to strictly adhere to provided instructions, ensuring JSON structure and data integrity.

        Guidelines:
        - DO NOT add, remove, or modify keys in the JSON object.
        - Replace placeholders with relevant data or leave them unchanged if no suitable data exists.
        - DO NOT add any commentrary, explanation, etc while replacing the placeholders. Use just the information.
        - Follow the real examples(if provided) to get idea what kind of information should be used to populate the keys of the object.
        - Maintain data types and adhere to the exact JSON format.
        - DO NOT change types. If the given object is a LIST, the returned object also MUST be a LIST, and so on.
        - Use the given prior context (traversal path) to better understand the need. The whole traversal path gives you a good context about what exactly the intended information for the given key should be. Thoughtfully use this context.
        - Produce valid JSON outputs without extraneous text or explanations.
        """

    user_prompt = f"""
        You are provided with the following inputs:
        1. A JSON object or a list of JSON objects.
        2. A definition of the JSON object
        3. A natural language description of a derivatives contract.
        4. The prior context (traversal path) leading to this object.

        ### Your Task:
        Find values that are best suited for the keys in the provided JSON object from the provided derivatives contract. Use the key and prior context to accurately find the values. Follow the instructions below.

        ### Instructions (READ AND STRICTLY FOLLOW):

        1. **Preserve the structure of the JSON object**:
        -  DO NOT modify, add or remove any keys or structural elements. Your job is to only populate the given ones.

        2. **Populate the placeholders**:
        - Use relevant data from the derivatives contract to replace the placeholders of the keys.
        - DO NOT directly copy information into the placeholder without considering its intended purpose.
        - For the given key, identify the best suited information from the given contract description and populate the corresponding placeholder
            - To achieve this, you must use the prior context (traversal path) to understand what exactly is needed and then you must look for that information very carefully in the contract description.
            For example, if the traversal path has any kind of 'date' as a substring, you must know that the placeholder will be a date.
        - DO NOT add any commentrary, explanation, etc while replacing the placeholders. Use just the information.

        3. **Maintain the same datatype**:
        - If the given object or the placeholder of any key is an empty string (\"\"), replace it with a STRING derived from the contract description.
        - Or if it is a LIST, populate it with a LIST of items derived from the contract information (add as many items as necessary). The output MUST BE a LIST.
        - Ensure that the placeholder is replaced with data of the correct type and structure.
        
        4. **Leave placeholder unchanged if no data is available**:
        - If no highly relevant data exists for the given key, leave its placeholder as-is.

        5. **Populate "description"-only keys**:
        - If a key contains only "description", populate it with ENUMERATED VALUES based on the description

        6. **Final reminder about the output format**:
        - Return a valid JSON object keeping the structure exactly the same as the input
        - DO NOT include explanations, comments, or any extraneous text.

        ### Inputs:
        - The JSON Object (to be populated):
        {obj}

        - Definition of the JSON object:
        {obj_definition}
        
        - Derivatives Contract Description:
        {contract_description}

        - Prior Context (Path):
        {path}
        """
    
    if use_rag:
        rag_context = llm_handler.get_context_using_rag(f"Find objects from the knowledge base that are STRUCTURALLY highly similar to the following object  :\n\n{obj}")
        user_prompt += "\n\nFinally, and MOST IMPORTANTLY, below are some real examples that illustrate the type of information contained in similar objects. You MUST follow these examples when populating the provided JSON object: \n" + rag_context

    return system_prompt, user_prompt




def populate_object(llm_handler, obj, contract_description, path, use_rag=False):
    obj_definition = "No definition available"
    if isinstance(obj, dict):
        obj_definition = obj.get("description", "No definition available")
   
    sys_prompt, user_prompt = create_prompt(llm_handler, contract_description, obj_definition, path, obj, use_rag)

    cnt = 0
    guidance = ""
    while(True):
        llm_response = llm_handler.generate(guidance+user_prompt, system_prompt=sys_prompt)
        response = extract_json(llm_response)
        if response != None:
            return response
        print(f"Failed to parse LLM response for {path}: {llm_response}. Generating again!!")
        guidance = "In your previous response, you did not output a correct intended JSON (just the populated JSON that is given as input). Follow the instructions more carefully and generate again:\n\n"
        cnt += 1
        if cnt >= 3:
            return obj


def populate_key(llm_handler, key, placeholder, contract_description, path, use_rag=False):
    
    if key == "description":
        return placeholder

    sys_prompt = """
        - You are a highly accurate and rule-abiding assistant specializing in processing derivatives contract data and populating JSON templates. Your primary goal is to adhere strictly to provided instructions. 
        - DO NOT add any commentrary, explanation, or any extraneous text while replacing the placeholders. Use just the key information.
        - Use the given prior context (traversal path) to better understand the need. The whole traversal path gives you a good context about what exactly the intended information for the given key should be. Thoughtfully use this context.
        """
    
    user_prompt = f"""
        You are provided with the following inputs:
        1. A 'key' from a JSON object.
        2. A placeholder for the given key.
        3. Natural language description of a derivatives contract
        4. The prior context (traversal path) leading to this key.

        ### Your Task:
        Find a value that is best suited for the provided key from the provided derivatives contract. Use the key and prior context to accurately find the value. Follow the instructions below.

        ### Instructions (READ AND STRICTLY FOLLOW):

        2. **Populate the placeholders**:
        - Use relevant data from the derivatives contract to replace placeholder.
        - DO NOT directly copy information into the placeholder without considering its intended purpose.
        - For the given key, identify the best suited information from the given contract description and populate the corresponding placeholder
            - To achieve this, you must use the prior context (traversal path) to understand what exactly is needed and then you must look for that information very carefully in the contract description.
            For example, if the traversal path has any kind of 'date' as a substring, you must know that the placeholder will be a date.

        3. **Maintain the same datatype**:
        - If the placeholder is an empty string (\"\"), replace it with a STRING derived from the contract information.
        - Or if it is a LIST, populate it with a LIST of items derived from the contract information (add as many items as necessary). The output MUST BE a LIST.
        - Ensure that the placeholder is replaced with data of the correct type and structure.
        
        5. **Leave placeholder unchanged if no data is available**:
        - If no highly relevant data exists for the given key, leave its placeholder as-is.

        6. **Final reminder about the output format**:
        - DO NOT include explanations, comments, or any extraneous text.

        ### Inputs:
        - The 'key' form a JSON Object (to be populated):
        {key}

        - Placeholder:
        {placeholder}
        
        - Derivatives Contract Description:
        {contract_description}

        - Prior Context (Path):
        {path}
        """

    if use_rag:
        rag_context = llm_handler.get_context_using_rag(f"Find keys from the knowledge base that are same as the key  : {key}")
        user_prompt += "\n\nFinally, and MOST IMPORTANTLY, below are some real examples that illustrate the type of information contained in similar keys. You MUST follow these examples when populating the provided JSON key: \n" + rag_context

    response = llm_handler.generate(user_prompt, system_prompt=sys_prompt)

    return response


def populate_template(llm_handler, key, template, contract_description, path="", d=3, use_rag=False):
    """Recursively traverse and populate the JSON template."""
    max_depth_below = max_depth(template)
    if max_depth_below <= d:
        # print("\nPopulating ---> ", template, "\n")
        populated = None
        if isinstance(template, (dict, list)):
            populated = populate_object(llm_handler, template, contract_description, path, use_rag)
        else:
            populated = populate_key(llm_handler, key, template, contract_description, path, use_rag)

        return populated


    if isinstance(template, dict):
        for curr_key, value in template.items():
            if curr_key == 'party':
                populated = populate_object(llm_handler, {curr_key : value}, contract_description, path, use_rag)
                template[curr_key] = populated[curr_key]
                continue
            
            current_path = f"{path}.{curr_key}" if path else curr_key
            if isinstance(value, (dict, list)):
                template[curr_key] = populate_template(llm_handler, curr_key, value, contract_description, current_path, d, use_rag)
            else:
                template[curr_key] = populate_key(llm_handler, curr_key, value, contract_description, current_path, use_rag)
    elif isinstance(template, list):
        for i, item in enumerate(template):
            current_path = f"{path}[{i}]"
            if isinstance(item, (dict, list)):
                template[i] = populate_template(llm_handler, key, item, contract_description, current_path, d, use_rag)
    return template




test_data_dir = "../../../../Data_from_BofA/FIIR_Confirms_32"
non_test_data_dir = "../../../data/type_wise_cdm_samples/non_test_data"
contract_templates_dir = "../../cdm_schema/type_wise_templates"
generated_cdm_parent_folder = "../../../../Data_from_BofA/FIIR_Confirms_32/test_results_cdm"


depth_threshold = 6
curr_llm = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# curr_llm = "Qwen/Qwen3-30B-A3B-Instruct-2507"

generated_cdm_folder = curr_llm.split('/')[-1]


test_contracts = load_test_contracts(test_data_dir)

for contract_type, contracts in test_contracts.items():

    if contract_type != "InterestRate":
        continue

    print(f"Processing contracts of type: {contract_type}\n")

    llm_handler = LLMHandler(
        model_id=curr_llm,
        token="huggingface access token",
        temperature=0.05
    )

    rag_knowledge_base_path = non_test_data_dir+"/json/"+contract_type

    llm_handler.initialize_rag(
        documents_path=rag_knowledge_base_path,
        embed_model_name="BAAI/bge-small-en-v1.5",
        top_k=5,
        similarity_cutoff=0.6,
        chunk_size=256,
        overlap=10
    )

    with open(contract_templates_dir+"/template_"+contract_type+".json") as f:
        template = json.load(f)

    for contract in contracts:
        print("\nProcessing "+contract['id']+"...")
        print("-"*60)
        
        without_rag_path = os.path.join(generated_cdm_parent_folder, generated_cdm_folder, "without_rag", contract_type, contract['id'])
        with_rag_path = os.path.join(generated_cdm_parent_folder, generated_cdm_folder, "with_rag", contract_type, contract['id'])

        without_rag_path = without_rag_path.replace(".txt", ".json")
        with_rag_path = with_rag_path.replace(".txt", ".json")

        os.makedirs(os.path.dirname(without_rag_path), exist_ok=True)
        os.makedirs(os.path.dirname(with_rag_path), exist_ok=True)

        if not os.path.exists(without_rag_path):
            with open(contract_templates_dir+"/template_"+contract_type+".json") as f:
                template_cdm = json.load(f)
            cdm = clean_cdm(populate_template(llm_handler, "cdm_template", template_cdm, contract['description'], path="", d=depth_threshold, use_rag=False))
            with open(without_rag_path, "w") as f:
                json.dump(cdm, f, indent=4)
                print("CDM Saved in " + without_rag_path)
        else:
            print("!!! CDM already exists: " + without_rag_path)
        
        
        if not os.path.exists(with_rag_path):
            with open(contract_templates_dir+"/template_"+contract_type+".json") as f:
                template_cdm_rag = json.load(f)
            cdm_rag = clean_cdm(populate_template(llm_handler, "cdm_template", template_cdm_rag, contract['description'], path="", d=depth_threshold, use_rag=True))
            with open(with_rag_path, "w") as f:
                json.dump(cdm_rag, f, indent=4)
                print("CDM_RAG Saved in " + with_rag_path)
        else:
            print("CDM_RAG already exists: " + with_rag_path)

