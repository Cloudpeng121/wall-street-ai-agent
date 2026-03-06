"""
ingest_pipeline.py (Cloud-Native Edition)
=========================================
Directly streams SEC HTML financial reports from Azure Blob Storage, 
cleans the markup, chunks the text, and ingests them into ChromaDB.
"""

import os
import logging
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from markdownify import markdownify
from azure.storage.blob import BlobServiceClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONTAINER_NAME = "financial-reports"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "financial_reports"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 纯正英语大脑

def get_azure_client() -> BlobServiceClient:
    load_dotenv(override=True)
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")
    return BlobServiceClient.from_connection_string(conn_str)

def stream_and_parse_from_azure(blob_service_client: BlobServiceClient) -> list[Document]:
    """从 Azure 内存流式读取 HTML 并转换为 LangChain 文档对象"""
    logger.info(f"Connecting to Azure Container: {CONTAINER_NAME}...")
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    documents = []
    # 遍历容器内的所有文件
    blobs = container_client.list_blobs()
    
    for blob in blobs:
        if blob.name.endswith((".html", ".htm")):
            logger.info(f"📥 Downloading and parsing from Cloud: {blob.name}")
            blob_client = container_client.get_blob_client(blob)
            
            # 直接将数据下载到内存 (二进制)，不生成本地临时文件
            html_content = blob_client.download_blob().readall()
            
            # 引入 Markdownify (确保在文件开头 import markdownify)
            
            
            # 先用 BS4 去除掉无用的脚本和样式代码，减轻负担
            soup = BeautifulSoup(html_content, "lxml")
            for script in soup(["script", "style"]):
                script.extract()
                
            # 🚨 核心魔法：将带有表格的 HTML 直接转为精美的 Markdown
            clean_text = markdownify(str(soup), heading_style="ATX", strip=['a', 'img'])
            
            # 组装成大模型认识的 Document 对象，并打上来源标签
            doc = Document(
                page_content=clean_text, 
                metadata={"source": f"azure://{blob.name}"}
            )
            documents.append(doc)
            
    logger.info(f"Successfully parsed {len(documents)} core reports from Azure.")
    return documents

def main():
    try:
        # 1. 连接云端数据源
        azure_client = get_azure_client()
        
        # 2. 提取并清洗数据 (Extract & Transform)
        raw_documents = stream_and_parse_from_azure(azure_client)
        if not raw_documents:
            logger.error("No HTML files found in Azure container! Aborting.")
            return

        # 3. 文本切片 (因为财报极长，需要切成适合模型消化的小块)
        logger.info("Splitting text into optimal RAG chunks...")
        # Increase chunk size to keep financial tables intact with their headers
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], # Prioritize splitting by paragraphs/blocks
            chunk_size=4000,  # Massively increase chunk size
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(raw_documents)
        logger.info(f"Created {len(chunks)} chunks ready for embedding.")

        # 4. 加载嵌入模型
        logger.info(f"Loading Embedding Engine: {EMBED_MODEL_NAME}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME, 
            model_kwargs={"device": "cpu"}
        )

        # 5. 写入向量数据库 (Load)
        logger.info("Building ChromaDB Vector Store...")
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR
        )

        logger.info("=" * 60)
        logger.info("✅ Cloud-to-Vector Pipeline Executed Successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()