#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证生成的tokenizer.json文件格式
"""

import json
import os

def validate_tokenizer_json(json_file: str):
    """验证tokenizer.json文件格式"""
    print(f"验证文件: {json_file}")
    
    if not os.path.exists(json_file):
        print(f"错误: 文件不存在 {json_file}")
        return False
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            tokenizer_dict = json.load(f)
        
        print(f"成功加载JSON文件，包含 {len(tokenizer_dict)} 个token")
        
        # 验证第一个token是特殊token
        if "0" in tokenizer_dict:
            token_0 = tokenizer_dict["0"]
            if isinstance(token_0, str) and "<|rwkv_tokenizer_end_of_text|>" in token_0:
                print("✓ Token 0 是正确的特殊token")
            else:
                print(f"✗ Token 0 格式错误: {token_0}")
                return False
        else:
            print("✗ 缺少Token 0")
            return False
        
        # 验证前几个普通token
        for i in range(1, min(10, len(tokenizer_dict))):
            token_key = str(i)
            if token_key in tokenizer_dict:
                token_value = tokenizer_dict[token_key]
                if isinstance(token_value, list) and len(token_value) > 0:
                    print(f"✓ Token {i}: {token_value}")
                else:
                    print(f"✗ Token {i} 格式错误: {token_value}")
                    return False
            else:
                print(f"✗ 缺少Token {i}")
                return False
        
        # 检查一些多字节token
        test_keys = ["257", "300", "500", "1000"]
        for key in test_keys:
            if key in tokenizer_dict:
                token_value = tokenizer_dict[key]
                if isinstance(token_value, list) and len(token_value) >= 2:
                    print(f"✓ 多字节Token {key}: {token_value}")
                else:
                    print(f"✗ 多字节Token {key} 格式错误: {token_value}")
        
        print("\n验证完成！tokenizer.json格式正确")
        return True
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return False
    except Exception as e:
        print(f"验证过程中出错: {e}")
        return False

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "..", "assets", "model", "tokenizer.json")
    json_file = os.path.normpath(json_file)
    
    validate_tokenizer_json(json_file)