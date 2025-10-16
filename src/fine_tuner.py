import os
import json
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from transformers import TrainingArguments

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

class FineTuner:
    def __init__(self, model_id, token, max_seq_length = 4096, dtype = None, load_in_4bit = True, json_folder="../data/type_wise_cdm_samples/non_test_data/json", txt_folder="../data/type_wise_cdm_samples/non_test_data/text"):
        self.model_id = model_id
        self.json_folder = json_folder
        self.txt_folder = txt_folder
        self.max_seq_length = 4096
        self.dtype = None
        self.load_in_4bit = True
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                                        model_name = self.model_id,
                                        max_seq_length = max_seq_length,
                                        dtype = dtype,
                                        load_in_4bit = load_in_4bit,
                                        token=token,
                                )
        print(model_id + " initialized for fine-tuning!")

    def convert_large_ints(self, data):
        """Recursively converts large integers to strings in dictionaries."""
        if isinstance(data, dict):
            return {k: self.convert_large_ints(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_large_ints(item) for item in data]
        elif isinstance(data, int) and abs(data) > 2**31 - 1:  # Check for large integer
            return str(data)
        return data

    def create_dataset(self):
        """Recursively loads JSON and TXT files from nested folders and formats them into a dataset."""
        data = []
        
        for root, _, files in os.walk(self.json_folder):  # Walk through all directories in json_folder
            for json_filename in files:
                if json_filename.endswith(".json"):
                    json_filepath = os.path.join(root, json_filename)
                    
                    # Load JSON data
                    with open(json_filepath, "r") as json_file:
                        json_data = json.load(json_file)
                    
                    # Convert large integers in JSON data
                    json_data = self.convert_large_ints(json_data)
                    
                    # Derive the corresponding TXT file path by mirroring the folder structure
                    relative_path = os.path.relpath(json_filepath, self.json_folder)  # Get relative path from json_folder
                    txt_filepath = os.path.join(self.txt_folder, relative_path.replace(".json", ".txt"))  # Locate the corresponding TXT file
                    
                    # Load TXT data if the file exists
                    if os.path.exists(txt_filepath):
                        with open(txt_filepath, "r") as txt_file:
                            txt_data = txt_file.read().strip()
                        
                        data.append({
                            "instruction": "Create an ISDA Common-Domain-Model (CDM) representation for the following OTC derivatives contract description : \n\n" + txt_data,
                            "output": json.dumps(json_data)  # Convert JSON to a string
                        })

        dataset = Dataset.from_pandas(pd.DataFrame(data))

        # Define the prompt template and EOS token
        alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Response:
        {}"""

        EOS_TOKEN = self.tokenizer.eos_token  # Ensure the EOS_TOKEN is set

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            outputs = examples["output"]
            texts = []
            for instruction, output in zip(instructions, outputs):
                # Add EOS_TOKEN to prevent infinite generation
                text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
                texts.append(text)
            return {"text": texts}

        # Apply the formatting function to the dataset
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        return dataset

    def fine_tune(self, max_steps=60):
        """Fine-tune the model using the dataset."""
    
        # Apply PEFT (Parameter-Efficient Fine-Tuning)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # Rank for LoRA
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )

        # Create the dataset
        dataset = self.create_dataset()
        print("Dataset Preparation Completed!")
        print("Training...")
        # Define training arguments
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=max_steps,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none"
            ),
        )

        # Train the model
        trainer_stats = trainer.train()

        print("Training Completed!")

    def generate(self, prompt):
        """Generate a response from the fine-tuned model."""
        alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Response:
        {}"""

        # Enable faster inference
        FastLanguageModel.for_inference(self.model)
        
        inputs = self.tokenizer(
            [alpaca_prompt.format(prompt, "")],  # Leave output blank for generation
            return_tensors="pt"
        ).to("cuda")

        # Generate the output from the model
        outputs = self.model.generate(**inputs, max_new_tokens=4096, use_cache=True)
        outputs = self.tokenizer.batch_decode(outputs)
        response_only = outputs[0].split("### Response:")[1].strip()
        
        return response_only


    def initialize_rag(self, documents_path: str, embed_model_name: str, top_k: int = 5, similarity_cutoff: float = 0.6):
        """
        Initialize the RAG (Retrieval-Augmented Generation) components, including embeddings, retrievers, and query engine.
        """
        self.top_k = top_k
        self.similarity_cutoff = similarity_cutoff

        Settings.llm = None
        Settings.chunk_size = 512
        Settings.overlap = 20
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # Load documents and create index
        documents = SimpleDirectoryReader(documents_path, recursive=True).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

        # Initialize retriever and query engine
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        self.rag_query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )
        print(f"Initialized RAG with {len(documents)} documents and embedding model: {embed_model_name}")

    def get_context_using_rag(self, query_prompt: str) -> str:
        """
        Retrieve relevant context using RAG and combine it with the query prompt for a more informed response.
        """
        if self.rag_query_engine is None:
            raise ValueError("RAG is not initialized. Please call `initialize_rag` before using this method.")

        # Retrieve examples
        retrieved_examples = self.rag_query_engine.query(query_prompt)
        
        # Build context
        context = ""
        for i in range(min(self.top_k, len(retrieved_examples.source_nodes))):
            context += retrieved_examples.source_nodes[i].text + "\n\n"

        return context
