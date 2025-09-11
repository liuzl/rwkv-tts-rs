#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将rwkv_vocab_v20230424.txt转换为JSON格式的tokenizer.json
格式：行号 'token内容' 长度 -> JSON数组格式
"""

import json
import re
import ast

def parse_vocab_line(line):
    """解析词表行：行号 'token内容' 长度"""
    line = line.strip()
    if not line:
        return None
    
    # 分割行：行号 token部分 长度
    parts = line.split(' ', 1)
    if len(parts) != 2:
        print(f"警告：无法解析行: {line}")
        return None
    
    line_num_str, rest = parts
    try:
        line_num = int(line_num_str)
    except ValueError:
        print(f"警告：无法解析行号: {line}")
        return None
    
    # 从右边找到最后一个空格，分离长度
    last_space = rest.rfind(' ')
    if last_space == -1:
        print(f"警告：无法找到长度: {line}")
        return None
    
    token_part = rest[:last_space]
    length_str = rest[last_space + 1:]
    
    try:
        length = int(length_str)
    except ValueError:
        print(f"警告：无法解析长度: {line}")
        return None
    
    # 提取token内容（去掉外层引号）
    if len(token_part) >= 2:
        if (token_part.startswith("'") and token_part.endswith("'")) or \
           (token_part.startswith('"') and token_part.endswith('"')):
            token_str = token_part[1:-1]
        elif token_part.startswith("b'") and token_part.endswith("'"):
            # 处理 b'\x...' 格式
            token_str = token_part[2:-1]  # 去掉 b' 和 '
        else:
            print(f"警告：token格式错误: {line}")
            return None
    else:
        print(f"警告：token太短: {line}")
        return None
    
    return line_num, token_str, length

def convert_token_to_bytes(token_str):
    """将token字符串转换为字节数组"""
    try:
        # 处理 \xAB\xCD 格式的字节序列
        if '\\x' in token_str:
            # 分割并解析十六进制字节
            parts = token_str.split('\\x')
            if len(parts) > 1:
                bytes_list = []
                for i, part in enumerate(parts):
                    if i == 0:
                        # 第一部分可能是空字符串或前缀
                        if part:
                            bytes_list.extend(list(part.encode('utf-8')))
                    else:
                        # 提取前两个字符作为十六进制
                        if len(part) >= 2:
                            hex_value = part[:2]
                            try:
                                byte_val = int(hex_value, 16)
                                bytes_list.append(byte_val)
                                # 如果还有剩余字符，继续处理
                                if len(part) > 2:
                                    bytes_list.extend(list(part[2:].encode('utf-8')))
                            except ValueError:
                                # 如果不是有效的十六进制，当作普通字符处理
                                bytes_list.extend(list(('\\x' + part).encode('utf-8')))
                        else:
                            bytes_list.extend(list(('\\x' + part).encode('utf-8')))
                return bytes_list
        
        # 处理单个转义序列
        if token_str.startswith('\\x') and len(token_str) == 4:
            # 十六进制转义序列如 \x00
            hex_value = token_str[2:]
            if len(hex_value) == 2:
                return [int(hex_value, 16)]
        
        # 处理特殊字符
        if token_str == '\\t':
            return [9]  # tab
        elif token_str == '\\n':
            return [10]  # newline
        elif token_str == '\\r':
            return [13]  # carriage return
        elif token_str == '\\\\':
            return [92]  # backslash
        elif token_str == "\\'": 
            return [39]  # single quote
        elif token_str == '\\"':
            return [34]  # double quote
        
        # 对于普通字符串，直接编码为UTF-8字节
        return list(token_str.encode('utf-8'))
        
    except Exception as e:
        print(f"警告：转换token失败 '{token_str}': {e}")
        # 回退：直接编码为UTF-8
        return list(token_str.encode('utf-8'))

def main():
    input_file = 'rwkv_vocab_v20230424.txt'
    output_file = '../assets/model/tokenizer.json'
    
    print(f"读取词表文件: {input_file}")
    
    tokens = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总行数: {len(lines)}")
    
    # 第一个token (ID 0) 设为特殊token
    tokens.append("<|rwkv_tokenizer_end_of_text|>")
    
    # 处理其余token (ID 1开始)
    for i, line in enumerate(lines):
        result = parse_vocab_line(line)
        if result is None:
            continue
            
        line_num, token_str, length = result
        
        # 转换为字节数组
        byte_array = convert_token_to_bytes(token_str)
        tokens.append(byte_array)
        
        # 打印进度
        if (i + 1) % 10000 == 0:
            print(f"已处理: {i + 1}/{len(lines)} 行")
    
    print(f"\n生成的token总数: {len(tokens)}")
    
    # 验证关键token
    print("\n验证关键token:")
    print(f"Token 0: {tokens[0]}")
    if len(tokens) > 1:
        print(f"Token 1: {tokens[1]}")
    if len(tokens) > 10464:
        print(f"Token 10464 (应该是'你'): {tokens[10464]}")
    if len(tokens) > 1000:
        print(f"Token 1000: {tokens[1000]}")
    
    # 生成JSON文件（格式化输出）
    print(f"\n生成JSON文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)
    
    print("转换完成！")
    
    # 验证生成的JSON文件
    print("\n验证生成的JSON文件...")
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_tokens = json.load(f)
    
    print(f"加载的token数量: {len(loaded_tokens)}")
    print(f"Token 0: {loaded_tokens[0]}")
    if len(loaded_tokens) > 10464:
        print(f"Token 10464: {loaded_tokens[10464]}")

if __name__ == '__main__':
    main()