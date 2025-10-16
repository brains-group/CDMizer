import sys
import os
import json
import signal

src_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(src_directory)

from llm_handler import *
from evaluate import *


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
                            "id": text_file.replace("txt", "json"),
                            "description": text_content
                        })
            
            if contracts:
                test_contracts[sub_folder] = contracts

    return test_contracts


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def get_coverage_with_timeout(llm_handler, description, cdm, TIMEOUT_SECONDS):
    """Runs get_coverage_score() with a timeout. If it exceeds the limit, returns None."""
    
    # Set up the timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)  # Start the timeout
    
    try:
        result = get_coverage_score(get_coverage(llm_handler, description, cdm))
        if result['score'] == 'undefined':
            return None
        signal.alarm(0)  # Cancel the alarm if successful
        return result
    except TimeoutException:
        return None



def generate_eval_results(llm_handler, test_contracts, generated_cdm_parent_folder, generated_cdm_folder, TIMEOUT_SECONDS = 900):

    for contract_type, contracts in test_contracts.items():
        
        if contract_type != "InterestRate":
            continue
        
        print("\n\n")
        print("=" * 60)
        print(f"Evaluating contracts of type: {contract_type}")
        print("=" * 60)

        for contract in contracts:
            print("\nEvaluating " + contract['id'] + "...")
            print("-" * 80)

            without_rag_path = os.path.join(generated_cdm_parent_folder, generated_cdm_folder, "without_rag", contract_type, contract['id'])
            with_rag_path = os.path.join(generated_cdm_parent_folder, generated_cdm_folder, "with_rag", contract_type, contract['id'])

            eval_without_rag_path = without_rag_path.replace("test_results_cdm", "evaluation_results_cdm")
            eval_with_rag_path = with_rag_path.replace("test_results_cdm", "evaluation_results_cdm")

            os.makedirs(os.path.dirname(eval_without_rag_path), exist_ok=True)
            os.makedirs(os.path.dirname(eval_with_rag_path), exist_ok=True)

            cdm, cdm_rag = {}, {}

            if os.path.exists(without_rag_path):
                with open(without_rag_path) as f:
                    cdm = json.load(f)
            else:
                print("!!! CDM does not exist: " + without_rag_path)

            if os.path.exists(with_rag_path):
                with open(with_rag_path) as f:
                    cdm_rag = json.load(f)
            else:
                print("!!! CDM_RAG does not exist: " + with_rag_path)

            if not os.path.exists(eval_without_rag_path):
                semantic_coverage = get_coverage_with_timeout(llm_handler, contract['description'], cdm, TIMEOUT_SECONDS)
                if semantic_coverage is not None:
                    print("Without RAG : ", semantic_coverage)
                    with open(eval_without_rag_path, "w") as f:
                        json.dump(semantic_coverage, f, indent=4)
                else:
                    print("!!!Skipping Without RAG evaluation due to timeout.")
            else:
                print("Without RAG : Results already generated!")

            if not os.path.exists(eval_with_rag_path):
                rag_semantic_coverage = get_coverage_with_timeout(llm_handler, contract['description'], cdm_rag, TIMEOUT_SECONDS)
                if rag_semantic_coverage is not None:
                    print("With RAG    : ", rag_semantic_coverage)
                    with open(eval_with_rag_path, "w") as f:
                        json.dump(rag_semantic_coverage, f, indent=4)
                else:
                    print("!!!Skipping With RAG evaluation due to timeout.")
            else:
                print("With RAG : Results already generated!")




# test_data_dir = "../../data/type_wise_cdm_samples/test_data"
test_data_dir = "../../../Data_from_BofA/FIIR_Confirms_32"

test_contracts = load_test_contracts(test_data_dir)

llm_handler = LLMHandler(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        token="huggingface access token",
        temperature=0.2
    )

generated_cdm_folder = "Meta-Llama-3.1-8B-Instruct"

TIMEOUT_SECONDS = 1200

generated_cdm_parent_folder = "../../../Data_from_BofA/FIIR_Confirms_32/test_results_cdm"
generate_eval_results(llm_handler, test_contracts, generated_cdm_parent_folder, generated_cdm_folder, TIMEOUT_SECONDS)