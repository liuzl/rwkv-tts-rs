#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def verify_tokenizer():
    print("Loading tokenizer.json...")
    with open('../assets/model/tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total tokens: {len(data)}")
    print("\nFormat verification:")
    print(f"Token 0 (special): {data['0']}")
    print(f"Token 1 (byte array): {data['1']}")
    print(f"Token 40 (byte array): {data['40']}")
    
    # Find the last token
    max_key = max(int(k) for k in data.keys())
    print(f"Token {max_key} (last): {data[str(max_key)]}")
    
    print("\nSample tokens:")
    for i in [100, 1000, 10000, 50000, 70000]:
        if str(i) in data:
            print(f"Token {i}: {data[str(i)]}")
        else:
            print(f"Token {i}: NOT FOUND")
    
    # Check token types
    string_tokens = 0
    array_tokens = 0
    for key, value in data.items():
        if isinstance(value, str):
            string_tokens += 1
        elif isinstance(value, list):
            array_tokens += 1
    
    print(f"\nToken type distribution:")
    print(f"String tokens: {string_tokens}")
    print(f"Array tokens: {array_tokens}")
    
    # Verify sequence
    missing_keys = []
    for i in range(max_key + 1):
        if str(i) not in data:
            missing_keys.append(i)
    
    if missing_keys:
        print(f"\nMissing token IDs: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    else:
        print("\nAll token IDs are present (0 to {}).".format(max_key))
    
    print("\nTokenizer verification complete!")

if __name__ == '__main__':
    verify_tokenizer()