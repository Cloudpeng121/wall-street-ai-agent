from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os

# 定义 DAG 的基础参数
default_args = {
    'owner': 'yunpeng', # 你的名字，作为这批任务的所有者
    'depends_on_past': False,
    'retries': 2, # 爬虫容易因为网络波动失败，设置 2 次自动重试
    'retry_delay': timedelta(minutes=5), # 每次重试间隔 5 分钟
}

@dag(
    dag_id='sec_10k_scraper_pipeline',
    default_args=default_args,
    description='Fetch SEC 10-K reports for Wall Street AI Agent',
    schedule_interval='@weekly', # 每周运行一次。如果是测试，可以在 UI 中手动触发
    start_date=datetime(2026, 1, 1),
    catchup=False, # 不要回补历史任务
    tags=['wall_street_agent', 'data_ingestion'],
)
def sec_scraper_pipeline():

    @task()
    def initialize_directories():
        """
        任务 1：确保数据存放的文件夹存在。
        注意：在 Docker 容器内部，我们的 dags 文件夹对应的是 /opt/airflow/dags
        """
        data_dir = '/opt/airflow/dags/data/sec_reports'
        os.makedirs(data_dir, exist_ok=True)
        print(f"Directory initialized at: {data_dir}")
        return data_dir

    @task()
    def fetch_sec_report(ticker: str, save_dir: str):
        """
        任务 2：执行 SEC 核心抓取逻辑
        """
        # 1. 极其重要的 Airflow 细节：将专属的 import 放在任务函数内部！
        # 这样可以避免 Airflow 调度器在解析整个 DAG 文件时因为找不到包而报错。
        from sec_edgar_downloader import Downloader
        import logging

        logger = logging.getLogger(__name__)
        print(f"🚀 开始抓取 {ticker} 的最新财报数据...")
        
        # 2. 完美复用你 edgar_to_azure.py 里的核心下载代码
        # 公司名和合规邮箱使用你原本的设定，保存路径使用 Airflow 动态分配的 save_dir
        dl = Downloader("MyDataProject", "yunpeng.wyp@gmail.com", save_dir)
        
        # 下载 10-Q
        logger.info(f"Fetching 10-Q for {ticker}...")
        dl.get("10-Q", ticker, limit=2, download_details=True)
        
        # 下载 10-K
        logger.info(f"Fetching 10-K for {ticker}...")
        dl.get("10-K", ticker, limit=1, download_details=True)
        
        print(f"✅ {ticker} 数据抓取成功！")
        
        # 返回公司代码 ticker，方便下一步的“上传 Azure 任务”或者“向量化任务”使用
        return ticker

    @task()
    def ingest_to_chroma(ticker: str, data_dir: str):
        """
        任务 3：完美复刻 ingest_pipeline.py 的核心清洗与向量化逻辑
        """
        import gc
        import logging
        from pathlib import Path
        from bs4 import BeautifulSoup
        from markdownify import markdownify
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        logger = logging.getLogger(__name__)
        print(f"🧠 开始处理 {ticker} 的数据并进行向量化...")

        # 定位刚才下载的文件夹
        target_folder = Path(data_dir) / "sec-edgar-filings" / ticker
        chroma_persist_directory = '/opt/airflow/dags/chroma_db'
        
        documents = []
        
        # 遍历寻找核心财报 html
        for root, _, files in os.walk(target_folder):
            for file in files:
                if file == "primary-document.html":
                    file_path = Path(root) / file
                    logger.info(f"📥 Parsing local file: {file_path}")
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    # 使用 BS4 去除无用标签
                    soup = BeautifulSoup(html_content, "lxml")
                    for script in soup(["script", "style"]):
                        script.extract()
                        
                    # 转化为精美 Markdown
                    clean_text = markdownify(str(soup), heading_style="ATX", strip=['a', 'img'])
                    
                    doc = Document(
                        page_content=clean_text, 
                        metadata={"source": f"local://{ticker}/{file_path.name}", "ticker": ticker}
                    )
                    documents.append(doc)

                    # 触发内存装甲
                    del html_content, soup, clean_text 
                    gc.collect()

        if not documents:
            logger.warning(f"⚠️ {ticker} 目录下未找到 primary-document.html")
            return f"{ticker}_skipped"

        # 文本切片
        logger.info("Splitting text into optimal RAG chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=4000,  
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks ready for embedding.")

        # 加载嵌入模型
        logger.info("Loading Embedding Engine: BAAI/bge-small-en-v1.5...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", 
            model_kwargs={"device": "cpu"}
        )

        # 写入向量数据库
        logger.info("Building ChromaDB Vector Store...")
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="financial_reports",
            persist_directory=chroma_persist_directory
        )

        logger.info(f"💾 {ticker} 的向量特征已成功 Upsert 至 ChromaDB！")
        return f"{ticker}_ingested"

    # --- 编排执行逻辑 (Workflow) ---
    
    # 1. 先执行初始化文件夹任务 (复用模板里的函数)
    save_directory = initialize_directories()
    
    # 2. 从你的代码里提取出的 7 大科技巨头
    target_tickers = ["MSFT", "TSLA", "NVDA", "GOOGL", "META", "AAPL", "AMZN"]
    
    # 3. 动态生成并行的抓取任务！
    for ticker in target_tickers:
        # fetch_sec_report 依赖于 initialize_directories 的输出
        fetch_sec_report(ticker=ticker, save_dir=save_directory)

# 实例化 DAG
dag_instance = sec_scraper_pipeline()
