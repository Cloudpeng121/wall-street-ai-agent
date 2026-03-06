# 📈 Wall Street AI Agent

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://python.langchain.com/)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat_LLM-purple.svg)](https://deepseek.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, cloud-native **Retrieval-Augmented Generation (RAG)** pipeline designed to ingest, process, and interactively query SEC financial reports (10-K, 10-Q) for companies like strictly **TSLA** and **MSFT** **e.g.**.

---

## 🚀 Overview

The **Wall Street AI Agent** is a full-stack AI system that bridges the gap between raw financial data from the SEC EDGAR database and an interactive LLM reasoning engine. It automates the extraction of dense financial tables and corporate text, structurally converts them to markdown, vectorizes the semantic chunks, and serves the data through an elegant, real-time chat interface powered by the deeply analytical **DeepSeek** model.

## 🏗️ System Architecture

The project employs a robust ETL (Extract, Transform, Load) and Query architecture. A key technical feature is the **iXBRL HTML-to-Markdown parsing strategy**, which uses `BeautifulSoup` and `Markdownify` to cleanly extract intricate financial tables from raw HTML, completely preserving the tabular structure before it enters the embedding model.

```mermaid
flowchart TD
    %% Define Node Styles
    classDef cloud fill:#0078D4,stroke:#fff,stroke-width:2px,color:#fff;
    classDef script fill:#2b3137,stroke:#fff,stroke-width:2px,color:#fff;
    classDef db fill:#00C7B7,stroke:#fff,stroke-width:2px,color:#fff;
    classDef llm fill:#4F46E5,stroke:#fff,stroke-width:2px,color:#fff;
    classDef ui fill:#FF4B4B,stroke:#fff,stroke-width:2px,color:#fff;

    subgraph Data Ingestion & Storage
        A[SEC EDGAR Database] -->|Fetch 10-K/10-Q| B(edgar_to_azure.py):::script
        B -->|Sync Raw HTML| C[(Azure Blob Storage)]:::cloud
    end

    subgraph Vectorization Pipeline
        C -->|Stream to Memory| D(ingest_pipeline.py):::script
        D -->|Clean & Extract Tables| E[Markdownify]
        E -->|Chunk Text| F[BAAI/bge-small Embeddings]
        F -->|Store Vectors| G[(ChromaDB)]:::db
    end

    subgraph Inference & UI
        G -->|Retrieve Top-K| H(app.py / LangChain):::script
        H <-->|Semantic Search & Prompt| I{DeepSeek LLM}:::llm
        H -->|Render Output| J[Streamlit Web App]:::ui
    end
```

## 🛠️ Tech Stack

**Ingestion & Cloud Storage:**
* `sec-edgar-downloader` (Targeted scraping of core financial docs)
* `azure-storage-blob` (Cloud Data Lake holding raw HTML)

**Data Processing & Vectorization:**
* `BeautifulSoup4` + `Markdownify` (HTML parsing and structural preservation)
* `LangChain` (Document loaders and text splitters)
* `HuggingFaceEmbeddings` (`BAAI/bge-small-en-v1.5`)
* `ChromaDB` (Local Vector Database)

**Frontend & Inference Engine:**
* `Streamlit` (Interactive Web App with Resource Caching)
* `ChatOpenAI` wrapper targeting the **DeepSeek API** (`deepseek-chat`)

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Clone this repository and navigate to the project root.

```bash
git clone https://github.com/Cloudpeng121/wall-street-ai-agent.git
cd wall-street-ai-agent
```

### 2. Environment Setup
Create a virtual environment (optional but recommended) and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 3. Configuration
Copy the `.env.example` file to create your own local `.env` file containing your secret keys.

```bash
cp .env.example .env
```
Fill in the following variables:
* `DEEPSEEK_API_KEY`: Your model inference key.
* `AZURE_STORAGE_CONNECTION_STRING`: Your Azure Blob storage connection string (used during the ingestion phase).

### 4. Running the Pipeline

**Step A: Ingest Data to Azure (Optional)**
*Extracts documents from SEC EDGAR and uploads core HTML pages to Azure Blob Storage.*
```bash
python src/edgar_to_azure.py
```

**Step B: Build Vector Database (Optional)**
*Streams HTML docs from Azure, converts to markdown, chunks, and builds the local `chroma_db`.*
```bash
python src/ingest_pipeline.py
```

**Step C: Launch the Web Agent (Main Entrypoint)**
*Starts the interactive chat interface.*
```bash
python -m streamlit run app.py
```

---

## 📸 App Preview

![Wall Street AI Agent UI](assets/ui_screenshot.png)

The app features a chat-based UI and simulated streaming responses directly grounded in validated real-world SEC source documentation.

---

## 🌟 Key Features

### 1. iXBRL HTML-to-Markdown Parsing Strategy
Raw SEC financial reports are notoriously messy, filled with thousands of lines of inline styles, scripts, and fragmented XBRL tags. Instead of attempting to use standard PDF OCR or raw text dumps, this pipeline targets the **HTML `primary-document`**.
By using `BeautifulSoup` to strip the garbage layers and passing the clean DOM tree into `Markdownify`, financial data tables (like Income Statements and Balance Sheets) are flawlessly converted into Markdown tables (`|---|---|---|`). This allows the DeepSeek LLM to mathematically reason over structured rows and columns rather than hallucinating over scrambled raw text.

### 2. Streamlit Resource Caching
The application UI wraps the heavy `ChromaDB` loading phase, the BGE embedding initialization, and the LangChain pipeline assembly inside a `@st.cache_resource` decorator. The UI re-renders instantly upon user chat interaction without paying the model loading startup tax twice.

### 3. Advanced Inference (DeepSeek)
Utilizes the high-intelligence `deepseek-chat` model fed by a rigorously restricted system prompt, enforcing that the model relies *strictly* on semantic chunk context. If data is absent, the agent is forced to truthfully admit the lack of information over hallucinating financial figures.
