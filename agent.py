"""
agent.py
========
Production-grade Financial RAG Agent using Hybrid Search, Cross-Encoder Reranking,
and Gemini 1.5 Pro for advanced financial reasoning.

Pipeline stages:
1. Load ChromaDB vectors.
2. Initialize BM25 (Sparse) and Chroma (Dense) retrievers.
3. Combine them into an EnsembleRetriever (Hybrid Search).
4. Apply BAAI/bge-reranker-base to rerank top chunks.
5. Pass top-3 highly relevant chunks to Gemini 1.5 Pro for the final answer.
"""

import logging
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# 🚨 核心改动 1：引入兼容 DeepSeek 的 OpenAI 客户端
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "financial_reports"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# 🚨 核心改动 2：指定 DeepSeek 模型
LLM_MODEL_NAME = "deepseek-chat" 

def main():
    # 强制覆盖加载，防止终端缓存作祟
    load_dotenv(override=True)
    
    # 读取 DeepSeek 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is missing in .env file.")

    # 1. 加载本地向量库 (我们昨天用 LlamaParse 做好的精美数据都在这)
    logger.info("Loading ChromaDB Vector Store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": "cpu"})
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME, 
        persist_directory=CHROMA_DIR, 
        embedding_function=embeddings
    )
    # 设定召回 Top 10 文本块
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. 初始化 DeepSeek 大脑
    logger.info(f"Loading LLM Engine: {LLM_MODEL_NAME} ...")
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME, 
        api_key=api_key, 
        base_url="https://api.deepseek.com", # 🚨 核心改动 3：指向 DeepSeek 官方接口
        temperature=0.0 # 财务问答必须保持严谨，拒绝幻觉
    )

    # 3. 定义金融分析师专属 Prompt
    prompt = ChatPromptTemplate.from_template(
        "You are an elite financial analyst. Use ONLY the provided context to answer the question.\n"
        "If you find figures for specific quarters, periods, or segments that partially answer the question, provide them clearly.\n"
        "If the data is completely missing from the context, state 'I cannot find this information in the reports.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # 格式化函数：把文档拼起来，顺带打印引用来源
    def format_docs(docs):
        print("\n--- 🔍 Retrieved Sources ---")
        for i, doc in enumerate(docs):
            print(f"{i+1}. Source: {doc.metadata.get('source', 'Unknown')}")
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. LCEL 链式调用
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("=" * 60)
    logger.info("🚀 Financial Agent is ONLINE (Powered by DeepSeek). Type 'exit' to stop.")
    logger.info("=" * 60)

    # 5. 交互式对话
    while True:
        user_query = input("\n[You]: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query.strip():
            continue
            
        try:
            logger.info("Thinking...")
            # 触发完整推理流
            answer = rag_chain.invoke(user_query)
            print(f"\n[Agent]: {answer}\n")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()