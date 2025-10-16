import os
import PyPDF2
from llm_handler import *


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


def convert_cdm_to_contract_description(llm, cdm_json):

    template1 = extract_text_from_pdf('../data/term_sheets/jpmorgan_term_sheet.pdf')
    template2 = extract_text_from_pdf('../data/term_sheets/rbccapital_term_sheet.pdf')

    prompt = (
        f"The following contract is defined in ISDA CDM format:\n"
        f"CDM Representation: {cdm_json}\n\n"
        f"Based on the above information, generate a contract description that follows the general style and tone "
        f"of the following two templates (without copying them exactly). Use clear and formal language to "
        f"describe the contract:\n\n"
        f"Template 1: {template1}...\n\n"
        f"Template 2: {template2}...\n\n"
        f"Now generate a description of the contract from the given CDM representation. Make sure to be concise and cover all the information from the CDM."
        f"Do not include unnecessary things from the templates."
    )

    contract_description = llm.generate(prompt)
    
    return contract_description



if __name__ == "__main__":
   
    llm = LLMHandler(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        token="huggingface access token"
    )

    source_dir = '../data/type_wise_cdm_samples/non_test_data/json'
    destination_dir = '../data/type_wise_cdm_samples/non_test_data/text'

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            source_file_path = os.path.join(root, file)
            print(f"Processing: {source_file_path}")
            with open(source_file_path, 'r') as f:
                file_content = f.read()
                    
            relative_path = os.path.relpath(root, source_dir)
            destination_subdir = os.path.join(destination_dir, relative_path)
            os.makedirs(destination_subdir, exist_ok=True)
            
            destination_file_name = os.path.splitext(file)[0] + '.txt'
            destination_file_path = os.path.join(destination_subdir, destination_file_name)

            if os.path.exists(destination_file_path):
                with open(destination_file_path, 'r') as dest_file:
                    existing_content = dest_file.read()
                    if len(existing_content) > 1000:
                        print(f"Already generated: {destination_file_path}")
                        continue

            processed_content = convert_cdm_to_contract_description(llm, file_content)

            with open(destination_file_path, 'w') as f:
                f.write(processed_content)
                print(f"Saved: {destination_file_path}")
