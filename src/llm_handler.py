import transformers
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from peft import PeftModel, PeftConfig


class LLMHandler:
    def __init__(
        self,
        model_id: str,
        token: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
        use_peft: bool = False,
    ):
        """
        Initialize the LLMHandler with a HuggingFace pipeline for text generation,
        or manually for PEFT-adapted models.
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_peft = use_peft

        if use_peft:
            print(f"[Info] Detected PEFT adapter at {model_id}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
            base_model_path = PeftConfig.from_pretrained(model_id, token=token).base_model_name_or_path
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=token
            )
            self.model = PeftModel.from_pretrained(base_model, model_id, token=token)
            self.model.eval()
        else:
            self.generator_pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model_kwargs={"torch_dtype": torch_dtype},
                device_map=device_map,
                token=token,
                truncation=True
            )

        print(f"[Initialized] LLM pipeline using model: {model_id}")

    def generate(self, prompt: str, max_new_tokens: int = None, system_prompt: str = None) -> str:
        """
        Query the LLM with a given prompt and return the response.
        Optionally include a system prompt.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.use_peft:
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:
            outputs = self.generator_pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.generator_pipeline.tokenizer.eos_token_id
            )
            return outputs[0]["generated_text"][-1]["content"]

    def initialize_rag(self, documents_path: str, embed_model_name: str, top_k: int = 5, similarity_cutoff: float = 0.6, chunk_size: int = 512, overlap: int = 20):
        """
        Initialize the RAG components, including embeddings, retrievers, and query engine.
        """
        self.top_k = top_k
        self.similarity_cutoff = similarity_cutoff

        Settings.llm = None
        Settings.chunk_size = chunk_size
        Settings.overlap = overlap
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        documents = SimpleDirectoryReader(documents_path, recursive=True).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

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

        retrieved_examples = self.rag_query_engine.query(query_prompt)
        context = ""
        for i in range(min(self.top_k, len(retrieved_examples.source_nodes))):
            context += retrieved_examples.source_nodes[i].text + "\n\n"

        return context
