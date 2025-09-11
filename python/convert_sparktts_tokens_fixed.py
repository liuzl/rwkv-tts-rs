#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本的SparkTTS特殊token转换脚本
正确处理字节序列格式和普通字符串格式
"""

import json
import re
import ast

def parse_token_line(line):
    """
    解析token行，支持多种格式：
    1. 行号 'token' 长度
    2. 行号 "token" 长度  
    3. 行号 b'字节序列' 长度
    """
    line = line.strip()
    if not line:
        return None, None, None
    
    # 使用正则表达式匹配不同格式
    # 格式1: 行号 b'字节序列' 长度
    byte_pattern = r'^(\d+)\s+b\'([^\']*)\'\'?\s+(\d+)$'
    byte_match = re.match(byte_pattern, line)
    if byte_match:
        line_num = int(byte_match.group(1))
        byte_str = byte_match.group(2)
        length = int(byte_match.group(3))
        
        # 尝试解码字节序列
        try:
            # 使用codecs处理转义序列，将\xXX格式转换为实际字节
            import codecs
            byte_data = codecs.decode(byte_str, 'unicode_escape')
            # 如果是字符串，编码为bytes再解码为UTF-8
            if isinstance(byte_data, str):
                byte_data = byte_data.encode('latin-1')
            decoded_token = byte_data.decode('utf-8')
            return line_num, decoded_token, length
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
            try:
                # 备用方法：直接处理字节序列
                byte_data = codecs.decode(byte_str, 'unicode_escape')
                return line_num, byte_data, length
            except:
                # 如果无法解码，保持原始格式
                return line_num, f"b'{byte_str}'", length
    
    # 格式2: 行号 'token' 长度 或 行号 "token" 长度
    quote_pattern = r'^(\d+)\s+([\'\"])(.*)\2\s+(\d+)$'
    quote_match = re.match(quote_pattern, line)
    if quote_match:
        line_num = int(quote_match.group(1))
        quote_char = quote_match.group(2)
        token_content = quote_match.group(3)
        length = int(quote_match.group(4))
        
        # 处理转义字符
        try:
            if quote_char == "'":
                # 处理单引号字符串
                token = ast.literal_eval(f"'{token_content}'")
            else:
                # 处理双引号字符串
                token = ast.literal_eval(f'"{token_content}"')
            return line_num, token, length
        except (ValueError, SyntaxError):
            # 如果解析失败，直接使用原始内容
            return line_num, token_content, length
    
    # 格式3: 简单格式 行号 token 长度（无引号）
    simple_pattern = r'^(\d+)\s+([^\s]+)\s+(\d+)$'
    simple_match = re.match(simple_pattern, line)
    if simple_match:
        line_num = int(simple_match.group(1))
        token = simple_match.group(2)
        length = int(simple_match.group(3))
        return line_num, token, length
    
    print(f"警告：无法解析行格式: {line}")
    return None, None, None

def convert_sparktts_tokens():
    input_file = 'rwkv_vocab_v20230424_sparktts_spct_tokens.txt'
    output_file = 'rwkv_vocab_v20230424_sparktts_spct_tokens.json'
    
    token_dict = {}
    error_count = 0
    success_count = 0
    
    print(f"开始转换 {input_file} 到 {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                line_num, token, length = parse_token_line(line)
                
                if line_num is not None and token is not None:
                    # 使用行号作为键，token作为值（与rwkv_vocab_v20230424.json格式一致）
                    token_dict[str(line_num)] = token
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:  # 只显示前10个错误
                        print(f"第{line_idx}行解析失败: {line.strip()}")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
    
    # 写入JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(token_dict, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成！")
        print(f"成功转换: {success_count} 个token")
        print(f"解析失败: {error_count} 行")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"写入JSON文件时发生错误: {e}")

if __name__ == '__main__':
    convert_sparktts_tokens()