from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen-7B",  # 或你要的模型路径
    local_dir="./qwen-7b",      # 下载到本地目录
    local_dir_use_symlinks=False  # 避免软链接问题（可选）
)