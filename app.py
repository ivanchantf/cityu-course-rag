import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader
)
from llama_index.readers.llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.readers.file import PyMuPDFReader
# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="Hugging Face RAG", layout="wide")
st.title("CityU Bachelor Degree Assistant🎓")
# HF_TOKEN = ""
with st.sidebar:
    st.header("Backend Configuration")

    backend = st.selectbox("Choose LLM Backend", [ "Local (Ollama) qwen3","Local (Ollama) llama3"])#"OpenRouter", "Ollama Cloud"
    st.divider()
    # show_sources = st.checkbox("Show Source Nodes & Scores", value=True)

# --- 2. GLOBAL COMPONENTS (Load once) ---
@st.cache_resource
def load_embedding_and_reranker():
    with st.spinner("Loading local models..."):
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        reranker_model = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=3
        )
        return embed_model, reranker_model
    
# Fetch models from cache
embed_model, reranker = load_embedding_and_reranker()
# Set Global Settings
Settings.embed_model = embed_model


# --- 3. DATA INDEXING ---
# Choose where to save
PERSIST_DIR_VECTOR   = "./storage_vector"
PERSIST_DIR_SUMMARY  = "./storage_summary"
@st.cache_resource
def get_indexes():
    # VECTOR INDEX
    if os.path.exists(PERSIST_DIR_VECTOR):
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR_VECTOR
            )
            vector_index = load_index_from_storage(storage_context)
            st.info("Loaded existing vector index from disk")
        except Exception as e:
            st.warning(f"Failed to load vector index: {e}\nRebuilding...")
            vector_index = None
    else:
        vector_index = None
    # SUMMARY INDEX
    if os.path.exists(PERSIST_DIR_SUMMARY):
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR_SUMMARY
            )
            summary_index = load_index_from_storage(storage_context)
            st.info("Loaded existing summary index from disk")
        except Exception as e:
            st.warning(f"Failed to load summary index: {e}\nRebuilding...")
            summary_index = None
    else:
        summary_index = None
    # Build only what's missing
    if vector_index is None or summary_index is None:
        if not os.path.exists("./data"):
            st.error("Missing data folder.")
            st.stop()
        # parser = LlamaParse(
        #     api_key="",          # get free / paid key at cloud.llamaindex.ai
        #     result_type="markdown",     # or "text"
        #     verbose=True,
        # )
        # file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            "./data",
            recursive=True,
            file_extractor={".pdf": PyMuPDFReader()}
        ).load_data()
        st.write("Debug: loaded documents")
        for i, doc in enumerate(documents):
            text_preview = doc.text.replace("\n", " ").strip()
            if len(text_preview) < 40:
                st.warning(f"Document {i} is basically empty or unreadable:\n{text_preview}")
            elif any(c in text_preview for c in ["%PDF", "/Filter", "/FlateDecode"]):
                st.error(f"Document {i} contains raw PDF binary — parser failed!")
            else:
                st.info(f"Document {i} preview: {text_preview}…")

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=5,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        if vector_index is None:
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(persist_dir=PERSIST_DIR_VECTOR)
            st.success("Built & saved vector index")
        if summary_index is None:
            summary_index = SummaryIndex.from_documents(documents)   # or from nodes if you prefer
            summary_index.storage_context.persist(persist_dir=PERSIST_DIR_SUMMARY)
            st.success("Built & saved summary index")
    return vector_index, summary_index

vector_index, summary_index = get_indexes()
# --- 4. LLM SETUP ---

# OPENROUTER_API_KEY = "" # Or use os.getenv("OPENROUTER_API_KEY")
# OLLAMA_CLOUD_API_KEY = ""
# if backend == "Ollama Cloud":
#     Settings.llm = Ollama(
#         model="gpt-oss:20b-cloud", # Example of a cloud-only model
#         base_url="https://ollama.com",
#         request_timeout=300.0,
#         additional_kwargs={
#             "headers": {"Authorization": f"Bearer {OLLAMA_CLOUD_API_KEY}"}
#         }
#     )
# elif backend == "OpenRouter":
#     Settings.llm = OpenRouter(
#         api_key=OPENROUTER_API_KEY,
#         # Examples: "anthropic/claude-3-sonnet", "openai/gpt-4o", "meta-llama/llama-3-70b-instruct"
#         model="z-ai/glm-4.5-air:free",
#         temperature=0.1,
#         max_tokens=1024,
#         context_window=128000 # Most OpenRouter models have large windows
#     )

if backend == "Local (Ollama) qwen3":
    Settings.llm = Ollama(model="qwen3:latest", request_timeout=120.0)
elif backend == "Local (Ollama) llama3":
    Settings.llm = Ollama(model="llama3:latest", request_timeout=120.0)


# --- 5. ENGINES ---
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window"), reranker]
    ),
    description="Specific facts and data."
)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
    description="General summaries."
)
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool]
)


# --- 6. UI ---
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
if prompt := st.chat_input("Ask a question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            response = query_engine.query(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
        except Exception as e:
            if "JSON" in str(e) or "parsing" in str(e):
                st.info("Router failed. Falling back...")
                # This used to fail because reranker wasn't defined here
                fallback_engine = vector_index.as_query_engine(
                    similarity_top_k=5,
                    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window"), reranker]
                )
                response = fallback_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            else:
                st.error(f"Error: {e}")