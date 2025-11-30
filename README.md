Testttt update readme

# RAG for Technical Signal-Processing Documentation

This project implements a secure, on-premise conversational AI system using RAG to query technical documentation (signal-processing reports, specs, internal PDFs) while keeping all data fully local.

The goal is to reproduce a realistic production-style RAG pipeline:
- Local ingestion and indexing of sensitive documents
- BGE-based embeddings and adaptive chunking
- FAISS vector search for low-latency retrieval
- Local LLMs (Mistral / Llama) for answer generation
- Basic evaluation of retrieval quality and latency

---

## Main Features

- **Fully on-premise RAG**  
  All embedding, retrieval and generation run locally.

- **Document ingestion pipeline**  
  Parsing and cleaning of PDFs / text reports, with metadata extraction and simple structure analysis.

- **Adaptive chunking strategy**  
  Hybrid approach mixing fixed-size chunks with semantic-aware merging, to improve retrieval quality.

- **Dense retrieval with BGE embeddings**  
  Uses BAAI BGE models for both document and query embeddings, with cosine-similarity search.

- **FAISS-based vector index**  
  Efficient local similarity search with FAISS, persistent index on disk, and separation of vectors and metadata.

- **Local LLM integration**  
  Interface for running models like Mistral 7B / Llama 3 via local inference, with prompt templates adapted to RAG.

- **Evaluation tools**  
  Simple scripts and notebooks to benchmark retrieval quality and latency for different retriever–generator combinations.

---

## Architecture

The RAG pipeline follows these main steps:

1. **Ingestion**
   - Load raw documents from `data/raw/` (PDF, text, etc.).
   - Extract text and basic structure (pages, sections) in `src/ingestion/`.

2. **Preprocessing & Chunking**
   - Clean text.
   - Split into chunks using an adaptive strategy (size in tokens, overlaps, merging of short segments) in `src/chunking/`.

3. **Embeddings & Indexing**
   - Compute dense embeddings for chunks with BGE models in `src/embedding/`.
   - Build a FAISS index and store metadata in `data/index/`.

4. **Retrieval**
   - For an incoming query, compute its embedding.
   - Retrieve top-k chunks via FAISS, optionally apply a re-ranking step.
   - Implemented in `src/retrieval/`.

5. **Generation (RAG)**
   - Format a prompt combining the user query and retrieved context.
   - Call a local LLM (Mistral / Llama).
   - Apply simple rules to avoid hallucinations.

7. **Evaluation**
   - Offline evaluation of retrieval and answer quality in `src/evaluation/` and `notebooks/`

---

