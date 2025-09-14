import os
import hashlib
import sys

def calculate_md5(file_path):
    """计算单个文件的MD5值"""
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except (IOError, PermissionError) as e:
        print(f"错误: 无法读取文件 {file_path} - {str(e)}")
        return None

def process_directory(directory_path):
    """遍历目录并计算每个文件的MD5"""
    if not os.path.isdir(directory_path):
        print(f"错误: {directory_path} 不是有效目录")
        return

    print(f"{'文件路径':<50} {'MD5值':<32}")
    print("-" * 85)
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            md5 = calculate_md5(file_path)
            if md5:
                print(f"{file_path:<50} {md5:<32}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python md5_calculator.py <目录路径>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    process_directory(target_dir)
