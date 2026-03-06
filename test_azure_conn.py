import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# 1. 加载 .env 文件中的环境变量，保护你的密钥不被硬编码在代码里
load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not connection_string:
    raise ValueError("❌ 找不到连接字符串，请检查你的 .env 文件！")

# 2. 初始化客户端与容器名
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "financial-reports-raw"
local_file_name = "test_vibe_coding.txt"

# 3. 在本地生成一个测试文件
with open(local_file_name, "w", encoding="utf-8") as f:
    f.write("Hello Azure! 这是一个来自 Antigravity 协作生成的测试文件。")

# 4. 执行上传逻辑
try:
    print(f"🔗 正在连接到 Azure Storage 容器: '{container_name}'...")
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob=local_file_name)

    print(f"⬆️ 开始上传文件 {local_file_name}...")
    with open(local_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    print("🎉 上传成功！快去 Azure Portal 的 Container 里看看吧！")

except Exception as e:
    print(f"❌ 上传失败，请检查容器是否创建或网络连接。错误详情: {e}")

finally:
    # 5. 清理本地测试垃圾文件
    if os.path.exists(local_file_name):
        os.remove(local_file_name)
        print("🧹 本地临时文件已清理。")
