#!/bin/bash

# 远程服务器信息
remote_user="csp"
remote_host="192.168.3.26"
remote_path="/home/csp/Documents/Project/ApexFL/"
remote_save_path="/home/csp/Documents/Project/ApexFL/save/"

# 本地目录路径
local_directory="../ApexFL/"
local_save_directory="../ApexFL/save/"

# 帮助函数
display_usage() {
    echo "Usage: $0 <d | r>"
    echo "Options:"
    echo "  d   Sync from local to remote server"
    echo "  r   Sync from remote server to local"
}

# 根据参数选择同步方向
if [ "$1" == "d" ]; then
    # 从本地同步到远程服务器，忽略.git目录
    rsync -avz --exclude='.git' "$local_directory" "$remote_user@$remote_host:$remote_path"
elif [ "$1" == "r" ]; then
    # 从远程服务器的save目录同步到本地
    rsync -avz "$remote_user@$remote_host:$remote_save_path" "$local_save_directory"
else
    display_usage
    exit 1
fi
