# RAG-Project

# RAG for Technical Signal-Processing Documentation

This project implements a secure, on-premise conversational AI system using RAG to query technical documentation (e.g., signal-processing reports, specs, internal PDFs) while keeping all data fully local.

The goal is to reproduce a realistic production-style RAG pipeline:
- Local ingestion and indexing of sensitive documents
- BGE-based embeddings and adaptive chunking
- FAISS vector search for low-latency retrieval
- Local LLMs (Mistral / Llama) for answer generation
- Basic evaluation of retrieval quality and latency

---

## Main Features

- **Fully on-premise RAG**  
  No external API calls. All embedding, retrieval and generation run locally.

- **Document ingestion pipeline**  
  Parsing and cleaning of PDFs / text reports, with metadata extraction and simple structure analysis.

- **Adaptive chunking strategy**  
  Hybrid approach mixing fixed-size chunks with semantic-aware merging (short paragraphs are merged, long sections are split), to improve retrieval quality.

- **Dense retrieval with BGE embeddings**  
  Uses BAAI BGE models for both document and query embeddings, with cosine-similarity search.

- **FAISS-based vector index**  
  Efficient local similarity search with FAISS, persistent index on disk, and separation of vectors and metadata.

- **Local LLM integration**  
  Interface for running models like Mistral 7B / Llama 3 via local inference (e.g. `llama.cpp` / `llama-cpp-python`), with prompt templates adapted to RAG.

- **Evaluation tools**  
  Simple scripts and notebooks to benchmark retrieval quality (Recall@k, MRR) and latency for different retriever–generator combinations.

---

## High-Level Architecture

The RAG pipeline follows these main steps:

1. **Ingestion**
   - Load raw documents from `data/raw/` (PDF, text, etc.).
   - Extract text and basic structure (pages, sections) in `src/ingestion/`.

2. **Preprocessing & Chunking**
   - Clean text (remove headers/footers, artifacts).
   - Split into chunks using an adaptive strategy (size in tokens, overlaps, merging of short segments) in `src/chunking/`.

3. **Embeddings & Indexing**
   - Compute dense embeddings for chunks with BGE models in `src/embedding/`.
   - Build a FAISS index and store metadata (document id, section, page) in `data/index/`.

4. **Retrieval**
   - For an incoming query, compute its embedding.
   - Retrieve top-k chunks via FAISS, optionally apply a re-ranking step.
   - Implemented in `src/retrieval/`.

5. **Generation (RAG)**
   - Format a prompt combining the user query and retrieved context.
   - Call a local LLM (Mistral / Llama) via `src/llm/`.
   - Apply simple rules to avoid hallucinations (“say I don’t know” if context is insufficient).

6. **Serving**
   - Expose a minimal API (FastAPI) in `src/api/`.
   - Optionally a simple web UI (Streamlit / Gradio) in `src/ui/`.

7. **Evaluation**
   - Offline evaluation of retrieval and answer quality in `src/evaluation/` and `notebooks/`.

---

## Repository Structure

Planned directory layout:

```bash
secure-rag-signal-docs/
│
├── src/
│   ├── ingestion/        # Loading, parsing, cleaning documents
│   ├── chunking/         # Adaptive chunking logic
│   ├── embedding/        # BGE embedder and helpers
│   ├── retrieval/        # FAISS index + retrieval pipeline
│   ├── llm/              # Local LLM wrapper and generation logic
│   ├── api/              # FastAPI endpoints for querying the system
│   ├── ui/               # Optional Streamlit/Gradio app
│   └── evaluation/       # Evaluation utilities (metrics, experiments)
│
├── data/
│   ├── raw/              # Source documents (not tracked in git)
│   ├── processed/        # Cleaned & chunked versions
│   └── index/            # FAISS index + metadata
│
├── notebooks/
│   ├── 01_ingestion_exploration.ipynb
│   ├── 02_embedding_and_indexing.ipynb
│   ├── 03_retrieval_evaluation.ipynb
│   └── 04_end_to_end_demo.ipynb
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_llm_integration.py
│
├── app.py                # Entry point for running the API / demo
├── requirements.txt      # Python dependencies
└── README.md
