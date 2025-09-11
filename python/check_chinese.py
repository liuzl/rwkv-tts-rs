#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def check_chinese_in_vocab():
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f'Total lines in vocab.txt: {len(lines)}')
    print('\nSearching for Chinese characters...')
    
    chinese_lines = []
    for i, line in enumerate(lines):
        if chinese_pattern.search(line):
            chinese_lines.append((i, line.strip()))
    
    if chinese_lines:
        print(f'Found {len(chinese_lines)} lines with Chinese characters:')
        for line_num, content in chinese_lines[:10]:  # Show first 10
            print(f'Line {line_num}: {content}')
        if len(chinese_lines) > 10:
            print(f'... and {len(chinese_lines) - 10} more lines')
    else:
        print('No Chinese characters found in vocab.txt')
        
        # Let's check some sample lines to see what's actually in the file
        print('\nSample lines from vocab.txt:')
        for i in [0, 1000, 30000, 50000, 70000]:
            if i < len(lines):
                print(f'Line {i}: {lines[i].strip()}')

if __name__ == '__main__':
    check_chinese_in_vocab()