# Smart Encoding and Automation of Over-The-Counter Derivatives Contracts

*A modular pipeline for converting natural language OTC derivatives into executable smart contracts via Common-Domain-Model (CDM) representations and reinforcement learning.*

---

## Overview

This repository enables an end-to-end automation flow for OTC derivatives processing:

1. **Natural Language contract description → CDM representation**
2. **CDM representation → Optimized Smart Contracts** (via Reinforcement Learning)

It also supports synthetic data generation and contract template creation.

---

## ⚙️ Installation

> ✅ Recommended: Use **Conda** with Python `3.11.10`

```bash
# Step 1: Clone the repository
git clone https://github.com/smart-derivative-contracts/automating-otc-derivatives.git
cd automating-otc-derivatives

# Step 2: Create and activate conda environment
conda create -n <your_env_name> python=3.11.10
conda activate <your_env_name>

# Step 3: Install dependencies
# If any package fails, remove it from requirements.txt and try again
pip install -r requirements.txt
```

---

## 🧪 Key Components

- `src/natural_language_to_cdm/`  
  Contains experiments and models for converting contract descriptions to CDM format

- `src/cdm_to_smart_contract/`  
  Implements CDM-to-Solidity translation using reinforcement learning

- `generate_text_from_cdm.py`  
  Generates synthetic natural language descriptions from CDM data

- `create_contract_templates.py`  
  Generates six contract templates based on the CDM schema

---

## 🔁 NL → CDM Workflow

The figure below illustrates the overall pipeline (CDMizer) used for populating CDM templates using RAG-enhanced prompting and LLM inference:

![image](https://github.com/user-attachments/assets/d1153071-f2db-494e-a7e2-a27803ca77c7)

*CDMizer Workflow: Recursive traversal, governed by a depth threshold (d), selects substructures (e.g., assignedIdentifier)
where the deepest subtree has a depth ≤ d. This ensures manageable task sizes for efficiency and accuracy. Context-aware prompts,
incorporating object definitions, traversal paths, and RAG-retrieved examples, guide the LLM in populating fields, which are then validated.
Recursive traversal ensures the entire structure is systematically completed.*

---

## 📌 Notes

- Python version must be 3.11.10 for compatibility.
- If certain packages in `requirements.txt` fail to install, comment them out and retry.
- The repository is under active development.
---
