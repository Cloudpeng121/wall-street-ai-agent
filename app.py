"""
app.py
======
Streamlit frontend for the Financial RAG Agent.
Connects to the existing ChromaDB vector store and DeepSeek LLM
without modifying any of the core pipeline logic from agent.py.
"""

import os
import time
import logging

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Constants  (mirror of agent.py — do NOT change)
# ---------------------------------------------------------------------------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "financial_reports"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "deepseek-chat"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wall Street AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark-finance aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Gradient background ---- */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #091426 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.15);
    }

    /* ---- Chat messages ---- */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 170, 0.08);
        margin-bottom: 0.75rem;
        backdrop-filter: blur(10px);
    }

    /* ---- User avatar override color ---- */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(0, 212, 170, 0.08);
    }

    /* ---- Chat input ---- */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0, 212, 170, 0.3) !important;
        border-radius: 12px !important;
        color: #e0e6ed !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #00d4aa !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.15) !important;
    }

    /* ---- General text ---- */
    .stMarkdown, p, li, span {
        color: #c8d6e5;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: rgba(0, 212, 170, 0.06);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: #00d4aa !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #8899aa !important;
    }

    /* ---- Headers ---- */
    h1 { color: #ffffff !important; }
    h2, h3 { color: #00d4aa !important; }

    /* ---- Divider ---- */
    hr { border-color: rgba(0, 212, 170, 0.15) !important; }

    /* ---- Spinner / status ---- */
    .stSpinner > div {
        border-top-color: #00d4aa !important;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 212, 170, 0.3); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resource: embeddings + vectorstore + RAG chain
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """
    Initialise all heavy resources exactly once per server session.
    Mirrors the setup logic in agent.py — no logic changes.
    """
    load_dotenv(override=True)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is missing from the .env file.")

    # 1. Embeddings + ChromaDB
    logger.info("Loading embeddings and ChromaDB …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. DeepSeek LLM
    logger.info("Loading DeepSeek LLM …")
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.0,
    )

    # 3. Prompt  (identical to agent.py)
    prompt = ChatPromptTemplate.from_template(
        "You are an elite Wall Street financial analyst tracking the Magnificent 7 tech equities.\n"
        "Use ONLY the provided context to answer the question. \n"
        "CRITICAL INSTRUCTION: Pay STRICT ATTENTION to the company name in the user's query and the source metadata. "
        "Do NOT mix up revenue, metrics, or strategies between different companies (e.g., Apple vs. Amazon).\n"
        "If you cannot find the specific company's data in the context, state 'I cannot find this information for [Company Name] in the reports.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # 4. format_docs helper + LCEL chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain ready ✓")
    return rag_chain


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Wall Street AI Agent")
    st.markdown("---")
    st.markdown(
        """
        **Powered by:**
        - 🧠 DeepSeek Chat LLM
        - 🗂️ ChromaDB Vector Store
        - 🔍 BGE-Small Embeddings
        - ⚡ LangChain LCEL Pipeline
        """
    )
    st.sidebar.markdown("""
    **Coverage:**
    - 🔹 **AAPL** — Apple Inc.
    - 🔹 **AMZN** — Amazon.com
    - 🔹 **GOOGL** — Alphabet (Google)
    - 🔹 **META** — Meta Platforms
    - 🔹 **MSFT** — Microsoft
    - 🔹 **NVDA** — NVIDIA Corp.
    - 🔹 **TSLA** — Tesla Inc.
    """)

    # Metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Top-K Chunks", "10")
    with col2:
        st.metric("Temp.", "0.0")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='color:#4a6070;font-size:0.75rem;text-align:center;margin-top:1rem;'>"
        "For research purposes only.<br>Not financial advice.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align:center;padding:2rem 0 1rem 0;'>
        <h1 style='font-size:2.6rem;font-weight:700;
                   background:linear-gradient(90deg,#00d4aa,#00a8ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            📈 Wall Street AI Agent
        </h1>
        <p style='color:#8899aa;font-size:1rem;margin-top:0.25rem;'>
            Ask anything about <strong style="color:#00d4aa;">TSLA</strong> &amp;
            <strong style="color:#00a8ff;">MSFT</strong> SEC financial reports
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS = [
    "What was Tesla's total revenue in the most recent fiscal year?",
    "Compare Microsoft's cloud revenue growth YoY.",
    "What are the main risk factors disclosed by Tesla?",
    "How much did MSFT spend on R&D last year?",
]

if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    for i, prompt in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(prompt, key=f"example_{i}", use_container_width=True):
            st.session_state.setdefault("messages", [])
            st.session_state["prefill"] = prompt
            st.rerun()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🙋" if message["role"] == "user" else "📊"):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Load the RAG chain (cached — runs only once)
# ---------------------------------------------------------------------------

with st.spinner("⚙️ Initialising AI engine (first load only) …"):
    try:
        rag_chain = load_rag_chain()
    except EnvironmentError as env_err:
        st.error(f"🔑 Configuration Error: {env_err}")
        st.info("Please add your `DEEPSEEK_API_KEY` to the `.env` file and restart.")
        st.stop()

# ---------------------------------------------------------------------------
# Handle example-prompt prefill
# ---------------------------------------------------------------------------

prefill = st.session_state.pop("prefill", None)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_query = st.chat_input("Ask about TSLA or MSFT financials …") or prefill

if user_query:
    # Store & display the user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_query)

    # Generate the AI response
    with st.chat_message("assistant", avatar="📊"):
        response_placeholder = st.empty()

        with st.spinner("📂 Analyzing SEC filings …"):
            try:
                full_response = rag_chain.invoke(user_query)
            except Exception as exc:
                logger.error("RAG chain error: %s", exc)
                st.error(
                    f"⚠️ The AI engine encountered an error:\n\n`{exc}`\n\n"
                    "Please check your API key / network and try again."
                )
                st.stop()

        # Simulated streaming / typing effect
        displayed = ""
        for char in full_response:
            displayed += char
            response_placeholder.markdown(displayed + "▌")
            time.sleep(0.008)          # ~125 chars/sec — feels natural
        response_placeholder.markdown(displayed)   # final render (no cursor)

    # Persist assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
