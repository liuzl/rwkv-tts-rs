#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试字节序列解码
"""

import codecs

# 测试字节序列
test_sequences = [
    '\\xed\\x99',
    '\\xef\\xb8',
    '\\xef\\xbc',
    '\\xef\\xbd',
    '\\xef\\xbf',
    '\\xf0\\x9d',
    '\\xf0\\x9f'
]

for seq in test_sequences:
    print(f"原始序列: {seq}")
    
    try:
        # 方法1: 直接使用codecs
        decoded1 = codecs.decode(seq, 'unicode_escape')
        print(f"  codecs解码: {repr(decoded1)}")
        
        if isinstance(decoded1, str):
            try:
                utf8_decoded = decoded1.encode('latin-1').decode('utf-8')
                print(f"  UTF-8解码: {repr(utf8_decoded)}")
            except UnicodeDecodeError as e:
                print(f"  UTF-8解码失败: {e}")
    except Exception as e:
        print(f"  解码失败: {e}")
    
    print()