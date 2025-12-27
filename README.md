# Local RAG Pipeline for Technical Documentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

> **A production-inspired, fully on-premise Retrieval-Augmented Generation (RAG) system.**
> Designed to query sensitive technical specs and research papers without data ever leaving your local machine.

---

## Overview

This project implements a secure, modular RAG pipeline capable of ingesting complex PDF documentation (research papers, technical specs), indexing them efficiently, and generating grounded answers using local LLMs (Mistral / Llama 3).

### Key Goals
* **Privacy-First:** Zero data leakage. All embeddings and inference run locally.
* **Production-Grade Architecture:** Modular design separating ingestion, indexing, and generation.
* **Scientific Precision:** Optimized for dense technical content (e.g., Signal Processing, ML Research).
* **Measurable Quality:** Includes an "LLM-as-a-Judge" evaluation pipeline to benchmark performance.

---

## Architecture

The pipeline follows a strict modular flow:

1.  **Ingestion:** Automated fetching of papers via `arXiv API` (or local PDFs).
2.  **Processing:** Layout-aware parsing using `pdfminer.six`.
3.  **Chunking:** Adaptive sliding-window strategy (800 tokens + 100 overlap) to preserve semantic context.
4.  **Embedding:** Dense vector generation using **BAAI/bge-small-en-v1.5**.
5.  **Vector Store:** High-performance similarity search with **FAISS**.
6.  **Generation:** Context-aware inference using **Ollama** (Mistral 7B).

---

## Quick Start

### 1. Prerequisites
You need **Python 3.9+** and **[Ollama](https://ollama.com/)** installed.

```bash
# Clone the repository
git clone [https://github.com/ALI-AL-MARJANI/RAG-Project.git](https://github.com/ALI-AL-MARJANI/RAG-Project.git)
cd RAG-Project

# Install Python dependencies
pip install -r requirements.txt
# (Or manually: pip install requests pdfminer.six sentence-transformers faiss-cpu feedparser numpy)

# Pull the LLM (Run this in a separate terminal)
ollama pull mistral
ollama run mistral
