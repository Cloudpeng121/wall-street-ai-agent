"""
edgar_to_azure.py - V3 (Timeout Defense Version)
================================================
"""

import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_TICKERS = ["MSFT", "TSLA", "NVDA", "GOOGL", "META", "AAPL", "AMZN"]
AMOUNT_TO_DOWNLOAD = 2 
DOWNLOAD_DIR = Path("./sec_data")
CONTAINER_NAME = "financial-reports"

def get_azure_client() -> BlobServiceClient:
    load_dotenv(override=True)
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING missing.")
    
    # 🚨 防御机制 1：显式增加超时限制 (单位：秒)
    return BlobServiceClient.from_connection_string(
        conn_str, 
        connection_timeout=300, # 5分钟连接超时
        read_timeout=300        # 5分钟读取超时
    )

def fetch_from_sec():
    logger.info("Starting SEC EDGAR download...")
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    # 使用你的合规邮箱
    dl = Downloader("MyDataProject", "yunpeng.wyp@gmail.com", str(DOWNLOAD_DIR))
    
    for ticker in TARGET_TICKERS:
        logger.info(f"Fetching 10-Q for {ticker}...")
        dl.get("10-Q", ticker, limit=AMOUNT_TO_DOWNLOAD, download_details=True)
        logger.info(f"Fetching 10-K for {ticker}...")
        dl.get("10-K", ticker, limit=1, download_details=True)

def upload_to_azure(blob_service_client: BlobServiceClient):
    logger.info("Starting Azure upload phase...")
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    if not container_client.exists():
        container_client.create_container()

    upload_count = 0
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            # 🚨 狙击手模式：只上传官方提取出的核心主报表，无视那些巨大的 TXT 和几百个碎图片
            if file == "primary-document.html":
                file_path = Path(root) / file
                blob_name = str(file_path.relative_to(DOWNLOAD_DIR)).replace("\\", "/")
                
                blob_client = container_client.get_blob_client(blob_name)
                
                logger.info(f"Uploading core report: {blob_name} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, max_concurrency=4)
                    upload_count += 1
            else:
                # 跳过不需要的大文件
                continue
                    
    logger.info(f"Azure sync complete. {upload_count} core reports uploaded.")

def main():
    try:
        fetch_from_sec()
        azure_client = get_azure_client()
        upload_to_azure(azure_client)
        logger.info("✅ Ingestion Layer Success!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()