
from llama_index.core import SimpleDirectoryReader

from llm_handler import *


if __name__ == "__main__":

    rag_knowledge_base_path = "../data/cdm_samples_json/fpml-5-10/products/fx"
    contract_descriptions_path = '../data/old_fine_tuning_data/contract_descriptions'

    llm_handler = LLMHandler(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        token="huggingface access token"
    )

    # Initialize RAG within the LLMHandler
    llm_handler.initialize_rag(
        documents_path=rag_knowledge_base_path,
        embed_model_name="BAAI/bge-small-en-v1.5",
        top_k=5,
        similarity_cutoff=0.6
    )

    # Load contract descriptions
    contract_descriptions = SimpleDirectoryReader(contract_descriptions_path, recursive=True).load_data()

    for item in contract_descriptions:
        contract = item.text
        
        # RAG context retrieval
        rag_context = llm_handler.get_context_using_rag(f"Find CDM representations that best capture the information from a derivatives contract like the following:\n\n{contract}")

        # Prepare prompts
        basic_prompt = f"""
        Represent the following derivatives contract in a complete ISDA CDM JSON format, adhering strictly to the CDM schema. Ensure all key details are covered and follow the schema definitions exactly:

        Derivatives Contract:
        {contract}

        Please ensure:
        1. The output is **concise** but **complete**, following the ISDA CDM dictionary.
        2. **Only** provide the JSON representation, with no additional text or explanations.
        3. If the content exceeds the token limit, **truncate** appropriately while still providing a **valid** and complete JSON structure.
        
        Output the **final JSON** below.
        """

        rag_prompt = f"""
        Represent the following derivatives contract in a complete ISDA CDM JSON format, adhering strictly to the CDM schema. Ensure all key details are covered and follow the schema definitions exactly:

        Derivatives Contract:
        {contract}

        Relevant CDM Examples for Reference:
        {rag_context}

        Please ensure:
        1. The output is **concise** but **complete**, following the ISDA CDM dictionary.
        2. **Only** provide the JSON representation, with no additional text or explanations.
        3. If the content exceeds the token limit, **truncate** appropriately while still providing a **valid** and complete JSON structure.
        
        Output the **final JSON** below.
        """

        # Query LLM
        cdm_response = llm_handler.generate(basic_prompt)
        cdm_with_rag_response = llm_handler.generate(rag_prompt)

        # Print or process the responses
        print("CDM Response without RAG:")
        print(cdm_response)
        print("\nCDM Response with RAG:")
        print(cdm_with_rag_response)

        break  # Remove this to process all contracts
