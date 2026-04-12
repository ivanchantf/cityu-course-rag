"""
RAG QA Evaluation & Comparison Script
======================================
Measures: Index Size (MB), Inference Latency (s), RAGAS scores
Models:  Flan-T5-base, Llama3 8B, Llama3 4-bit, Qwen3 8B, Qwen3 4-bit

Prerequisites:
  pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface \
              llama-index-readers-file llama-index-postprocessor-sbert-rerank \
              ragas datasets transformers torch pandas tabulate psutil

  Ollama models pulled:
    ollama pull llama3:latest          # 8B
    ollama pull llama3:8b-instruct-q4_0 # 4-bit
    ollama pull qwen3:latest           # 8B
    ollama pull qwen3:8b-q4_K_M         # 4-bit
"""

import os, time, json, shutil, psutil, traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import PrivateAttr

# ---------- llama-index imports ----------
from llama_index.core import (
    VectorStoreIndex, SummaryIndex, Settings, StorageContext,
    load_index_from_storage, SimpleDirectoryReader,
)
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.llms import (
    CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

# ---------- RAGAS ----------
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

os.environ["OMP_NUM_THREADS"] = "1"

# ════════════════════════════════════════════════════════════════
# 1.  T5 LLM wrapper (same as app.py)
# ════════════════════════════════════════════════════════════════

class T5LLM(CustomLLM):
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 256
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, model_name="google/flan-t5-base", max_new_tokens=256, **kw):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens, **kw)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name, context_window=32768, num_output=self.max_new_tokens)

    @llm_completion_callback()
    def complete(self, prompt: str, **kw) -> CompletionResponse:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return CompletionResponse(text=self._tokenizer.decode(out[0], skip_special_tokens=True))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kw) -> CompletionResponseGen:
        r = self.complete(prompt, **kw)
        yield CompletionResponse(text=r.text, delta=r.text)


# ════════════════════════════════════════════════════════════════
# 2.  CONFIG
# ════════════════════════════════════════════════════════════════

DATA_DIR = "./data"                       # folder with CityU PDFs
RESULTS_FILE = "eval_results.json"
RESULTS_CSV  = "eval_results.csv"

# -- Evaluation questions (adapt to your domain) ----------------
# Each entry: (question, ground_truth_answer)
# ground_truth is needed for context_recall; make it a short factual sentence.
EVAL_QA_PAIRS: List[Dict] = [
    {
        "question": "What are the graduation requirements for a Bachelor of Science in Computer Science at CityU?",
        "ground_truth": "Students must complete 120 credit units including major, GE, and free elective requirements.",
    },
    {
        "question": "How many credit units are required for the GE courses?",
        "ground_truth": "Students need to complete 30 credit units of Gateway Education courses.",
    },
    {
        "question": "What is the medium of instruction at CityU?",
        "ground_truth": "The medium of instruction is English.",
    },
    {
        "question": "Can students take minor programmes?",
        "ground_truth": "Yes, students can declare a minor subject to departmental approval.",
    },
    {
        "question": "What is the maximum study period for a bachelor's degree at CityU?",
        "ground_truth": "The maximum study period is typically 8 years for a 4-year programme.",
    },
]

# -- Models to evaluate ------------------------------------------
MODEL_CONFIGS = [
    {"name": "T5-base",        "type": "t5",    "hf_model": "google/flan-t5-base"},
    {"name": "Llama3-8B",      "type": "ollama", "ollama_model": "llama3:latest"},
    {"name": "Llama3-4bit",    "type": "ollama", "ollama_model": "llama3:8b-instruct-q4_0"},
    {"name": "Qwen3-8B",       "type": "ollama", "ollama_model": "qwen3:latest"},
    {"name": "Qwen3-4bit",     "type": "ollama", "ollama_model": "qwen3:8b-q4_K_M"},
]

# -- Chunking strategies (shared across models) ------------------
CHUNKING_CONFIGS = [
    {
        "name": "SentenceWindow-5",
        "parser": lambda: SentenceWindowNodeParser.from_defaults(window_size=5),
        "use_window_replacement": True,
    },
    {
        "name": "SentenceSplitter-256",
        "parser": lambda: SentenceSplitter(chunk_size=256, chunk_overlap=32),
        "use_window_replacement": False,
    },
    {
        "name": "TokenSplitter-512",
        "parser": lambda: TokenTextSplitter(chunk_size=512, chunk_overlap=64),
        "use_window_replacement": False,
    },
]


# ════════════════════════════════════════════════════════════════
# 3.  HELPERS
# ════════════════════════════════════════════════════════════════

def dir_size_mb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 3)


def ram_usage_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)


def load_documents():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Missing '{DATA_DIR}' folder with PDFs.")
    return SimpleDirectoryReader(
        DATA_DIR, recursive=True, file_extractor={".pdf": PyMuPDFReader()}
    ).load_data()


def build_vector_index(documents, node_parser, persist_dir):
    """Build (or load) a vector index and return it + its disk size."""
    if os.path.exists(persist_dir):
        import gc, time
        gc.collect()
        for attempt in range(3):
            try:
                shutil.rmtree(persist_dir)
                break
            except PermissionError:
                time.sleep(1)
    nodes = node_parser.get_nodes_from_documents(documents)
    idx = VectorStoreIndex(nodes)
    idx.storage_context.persist(persist_dir=persist_dir)
    return idx, dir_size_mb(persist_dir), len(nodes)


def get_llm(cfg: Dict):
    if cfg["type"] == "t5":
        return T5LLM(model_name=cfg["hf_model"])
    else:
        return Ollama(model=cfg["ollama_model"], request_timeout=180.0)


def run_queries(query_engine, questions: List[str]):
    """Run all questions; return answers, contexts, total latency."""
    answers, contexts, latencies = [], [], []
    for q in questions:
        t0 = time.perf_counter()
        resp = query_engine.query(q)
        latencies.append(time.perf_counter() - t0)
        answers.append(resp.response or "")
        # collect retrieved context strings
        ctx = []
        if hasattr(resp, "source_nodes"):
            for sn in resp.source_nodes:
                ctx.append(sn.node.get_content())
        contexts.append(ctx)
    return answers, contexts, latencies


def compute_ragas(questions, answers, contexts, ground_truths):
    """Compute RAGAS metrics using Ollama as the judge LLM."""
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import HuggingFaceEmbeddings

    judge_llm = ChatOllama(model="llama3:latest", temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,        # list[list[str]]
        "ground_truth": ground_truths,
    })
    try:
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        result_df = result.to_pandas()
        scores = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in result_df.columns:
                scores[metric] = round(result_df[metric].mean(), 4)
            else:
                scores[metric] = None
        return scores
    except Exception as e:
        print(f"  ⚠ RAGAS evaluation failed: {e}")
        traceback.print_exc()
        return {"faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "context_recall": None}


# ════════════════════════════════════════════════════════════════
# 4.  MAIN EVALUATION LOOP
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  RAG Evaluation: Index Size | Latency | RAGAS")
    print("=" * 70)

    # Shared embedding & reranker
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
    Settings.embed_model = embed_model

    documents = load_documents()
    questions    = [p["question"] for p in EVAL_QA_PAIRS]
    ground_truths = [p["ground_truth"] for p in EVAL_QA_PAIRS]

    all_results = []

    idx = None
    for chunk_cfg in CHUNKING_CONFIGS:
        parser = chunk_cfg["parser"]()
        persist_dir = f"./storage_eval_{chunk_cfg['name']}"

        print(f"\n{'─'*60}")
        print(f"Chunking: {chunk_cfg['name']}")
        print(f"{'─'*60}")

        del idx
        import gc; gc.collect()
        idx, index_mb, num_nodes = build_vector_index(documents, parser, persist_dir)
        print(f"  Index size: {index_mb} MB  |  Nodes: {num_nodes}")

        for model_cfg in MODEL_CONFIGS:
            print(f"\n  ▸ Model: {model_cfg['name']}")
            ram_before = ram_usage_mb()

            try:
                llm = get_llm(model_cfg)
                Settings.llm = llm

                postprocessors = [reranker]
                if chunk_cfg["use_window_replacement"]:
                    postprocessors.insert(0, MetadataReplacementPostProcessor(target_metadata_key="window"))

                is_t5 = model_cfg["type"] == "t5"
                engine = idx.as_query_engine(
                    similarity_top_k=3 if is_t5 else 10,
                    node_postprocessors=postprocessors,
                )

                answers, contexts, latencies = run_queries(engine, questions)
                avg_latency = round(sum(latencies) / len(latencies), 3)
                total_latency = round(sum(latencies), 3)
                ram_after = ram_usage_mb()

                print(f"    Avg latency: {avg_latency}s  |  RAM delta: {ram_after - ram_before:.1f} MB")

                # RAGAS
                print("    Computing RAGAS scores...")
                ragas_scores = compute_ragas(questions, answers, contexts, ground_truths)
                print(f"    RAGAS: {ragas_scores}")

                row = {
                    "chunking": chunk_cfg["name"],
                    "model": model_cfg["name"],
                    "index_size_mb": index_mb,
                    "num_nodes": num_nodes,
                    "avg_latency_s": avg_latency,
                    "total_latency_s": total_latency,
                    "ram_before_mb": ram_before,
                    "ram_after_mb": ram_after,
                    **ragas_scores,
                    "answers": answers,
                }
                all_results.append(row)

            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                traceback.print_exc()
                all_results.append({
                    "chunking": chunk_cfg["name"],
                    "model": model_cfg["name"],
                    "error": str(e),
                })

    # ── Save results ──────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "answers"} for r in all_results])
    df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_FILE} and {RESULTS_CSV}")


if __name__ == "__main__":
    main()