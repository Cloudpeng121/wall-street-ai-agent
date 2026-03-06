# Wall Street AI Agent

A robust Retrieval-Augmented Generation (RAG) system for querying and analyzing SEC financial reports (TSLA & MSFT).

## Overview

This project implements a full LLM-based RAG pipeline to answer financial queries based on local SEC documents. It leverages:
- **LangChain** and **ChromaDB** for document ingestion, chunking, and embedding.
- **BAAI/bge-small-en-v1.5** via `HuggingFaceEmbeddings` for vector search.
- **DeepSeek Chat LLM** through `ChatOpenAI` as the core reasoning engine.
- A **Streamlit** Web Interface (`app.py`) for an interactive, conversational user experience.
- A fallback CLI Agent (`agent.py`) for quick command-line queries.
- Modular data ingestion scripts (`ingest_pipeline.py`) to process and handle raw PDFs automatically.

## Quick Start

1. Provide your environment variables across a `.env` file at the root:
```ini
DEEPSEEK_API_KEY="your-api-key-here"
```

2. (Optional) Run document ingestion if `chroma_db` is empty:
```bash
python ingest_pipeline.py
```

3. Launch the Streamlit web app:
```bash
python -m streamlit run app.py
```

## App Preview

The app features a chat-based UI and simulated streaming responses directly grounded in validated real-world SEC source documentation.
