# CityU Bachelor Degree Assistant 🎓

A Streamlit-based RAG chatbot using LlamaIndex and local Hugging Face / Ollama backends to answer questions over a collection of academic documents.

---

## 🏗️ Technical Highlights

* **Small-to-Big Retrieval**: Uses `SentenceWindowNodeParser` to index singular sentences for high-precision matching, then restores the surrounding paragraph context for the LLM.
* **Hybrid Search & Re-ranking**: Combines dense vector retrieval with a tree-structured summary index, re-scored via a semantic cross-encoder (`ms-marco-MiniLM-L-2-v2`).
* **Agentic Routing**: Features an intent-based router that automatically splits traffic—directing factual QA to the vector index and broad synthesis requests to the summary index.

---

## 🛠️ Quick Start

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running locally.

### 1. Install Dependencies
Ensure a `requirements.txt` file is present at the root level containing the following packages:

```text
streamlit>=1.30.0
llama-index-core>=0.10.0
pymupdf>=1.23.0
llama-index-embeddings-huggingface>=0.1.0
llama-index-postprocessor-sbert-rerank>=0.1.0
llama-index-llms-ollama>=0.1.0
pandas>=2.0.0
sentence-transformers
transformers
```
Install them via your terminal:

```Bash
pip install -r requirements.txt
```

### 2. Download Local LLM Models
Pull the required model weights using Ollama:

```Bash
ollama pull qwen3:latest
ollama pull llama3:latest
```

### 3. Place Documents & Run
Place your target academic PDFs inside folder named data/ in the project root.

Launch the application:

```Bash
streamlit run app.py
```
📌 Implementation Details

- Persistent Storage Caching: On the first run, the app builds and serializes the index files into ./storage_vector and ./storage_summary to enable fast, warm starts on subsequent launches.

- Fallback Strategy: If the agentic routing layer encounters an unexpected error, the interface seamlessly drops back to a standard token-restricted vector index query pipeline to maintain continuous uptime.

📂 Project Structure
```Plaintext
├── app.py              # Streamlit interface, pipeline setup, and routing logic
├── data/               # Source folder for project target PDFs
├── storage_vector/     # Cached vector search indices
└── storage_summary/    # Cached summary index hierarchies
```