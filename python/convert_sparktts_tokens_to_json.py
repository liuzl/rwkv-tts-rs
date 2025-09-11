#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 rwkv_vocab_v20230424_sparktts_spct_tokens.txt 转换为 JSON 格式的键值对
格式：行号 'token' 长度 -> {"token": 行号}
"""

import json
import re
import os

def parse_token_file(input_file, output_file):
    """
    解析token文件并转换为JSON格式
    
    Args:
        input_file: 输入的txt文件路径
        output_file: 输出的JSON文件路径
    """
    token_dict = {}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 使用正则表达式解析格式：行号 'token' 长度
                # 匹配模式：数字 空格 单引号 token内容 单引号 空格 数字
                match = re.match(r'^(\d+)\s+\'([^\']*)\' \d+$', line)
                if match:
                    line_number = int(match.group(1))
                    token = match.group(2)
                    
                    # 将token作为键，行号作为值
                    token_dict[token] = line_number
                else:
                    print(f"警告：第{line_num}行格式不匹配: {line}")
        
        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(token_dict, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成！")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"总共处理了 {len(token_dict)} 个token")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except Exception as e:
        print(f"错误：{str(e)}")

def main():
    # 文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'rwkv_vocab_v20230424_sparktts_spct_tokens.txt')
    output_file = os.path.join(script_dir, 'rwkv_vocab_v20230424_sparktts_spct_tokens.json')
    
    print("开始转换 SparkTTS 特殊token文件...")
    parse_token_file(input_file, output_file)

if __name__ == '__main__':
    main()