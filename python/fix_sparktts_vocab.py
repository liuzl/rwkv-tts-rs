#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re

def parse_sparktts_vocab():
    """
    修复SparkTTS词表转换问题：
    1. 正确的键值对顺序：token ID作为键，token内容作为值
    2. 处理特殊字符转换问题
    """
    input_file = 'rwkv_vocab_v20230424_sparktts_spct_tokens.txt'
    output_file = 'rwkv_vocab_v20230424_sparktts_spct_tokens.json'
    
    vocab_dict = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # 使用正则表达式解析行格式：行号 'token' 长度
            # 处理单引号和双引号的情况
            match = re.match(r'^(\d+)\s+(["\'])(.*)\2\s+(\d+)$', line)
            
            if match:
                token_id = match.group(1)
                quote_char = match.group(2)
                token_content = match.group(3)
                token_length = match.group(4)
                
                # 处理转义字符
                if quote_char == '"':
                    # 双引号内的内容，处理JSON转义
                    try:
                        # 先尝试作为JSON字符串解析
                        token_content = json.loads('"' + token_content + '"')
                    except json.JSONDecodeError:
                        # 如果解析失败，保持原样
                        pass
                else:
                    # 单引号内的内容，处理Python字符串转义
                    try:
                        # 处理常见的转义序列
                        token_content = token_content.encode().decode('unicode_escape')
                    except UnicodeDecodeError:
                        # 如果解析失败，保持原样
                        pass
                
                # 正确的格式：token ID作为键，token内容作为值
                vocab_dict[token_id] = token_content
                
            else:
                # 处理格式不匹配的行
                print(f"警告：第{line_num}行格式不匹配: {line}")
                
                # 尝试其他解析方式
                parts = line.split()
                if len(parts) >= 3:
                    token_id = parts[0]
                    # 合并中间部分作为token内容，去掉引号
                    token_content = ' '.join(parts[1:-1]).strip('"\'')
                    vocab_dict[token_id] = token_content
    
    # 写入JSON文件，确保格式正确
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总共处理了 {len(vocab_dict)} 个token")
    
    # 显示前几个和后几个条目作为验证
    print("\n前5个条目:")
    for i, (k, v) in enumerate(list(vocab_dict.items())[:5]):
        print(f'  "{k}": "{v}"')
    
    print("\n后5个条目:")
    for i, (k, v) in enumerate(list(vocab_dict.items())[-5:]):
        print(f'  "{k}": "{v}"')

if __name__ == '__main__':
    parse_sparktts_vocab()