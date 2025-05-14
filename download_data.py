from datasets import load_dataset, DownloadConfig

download_config = DownloadConfig(
    cache_dir="./data",    # 存放下载和解压文件
    #download_dir="/data_download", # 存放原始下载包
    #extract_dir="/data"   # 存放解压后文件
)

dataset = load_dataset(
    "cnn_dailymail",
    "3.0.0",
    download_config=download_config
)

print(dataset)