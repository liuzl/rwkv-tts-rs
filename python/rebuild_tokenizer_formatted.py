#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import ast

def rebuild_tokenizer_formatted():
    print("Reading vocab.txt...")
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} tokens...")
    
    tokenizer_data = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Token ID 0 is special - keep as string
        if i == 0:
            tokenizer_data[str(i)] = line
            print(f"Token {i}: {line} (special token)")
        else:
            # For other tokens, convert from byte string to array
            if line.startswith("b'") and line.endswith("'"):
                try:
                    # Parse the byte string
                    byte_str = ast.literal_eval(line)
                    if isinstance(byte_str, bytes):
                        # Convert bytes to list of integers
                        byte_array = list(byte_str)
                        tokenizer_data[str(i)] = byte_array
                    else:
                        # If it's a string, convert to bytes first
                        byte_array = list(byte_str.encode('utf-8'))
                        tokenizer_data[str(i)] = byte_array
                except Exception as e:
                    # If parsing fails, try manual parsing
                    if line == "b'\"'":
                        # Special case for quote character
                        tokenizer_data[str(i)] = [34]  # ASCII for "
                    elif line == "b'\''":
                        # Special case for single quote
                        tokenizer_data[str(i)] = [39]  # ASCII for '
                    else:
                        # Try to extract the content between b' and '
                        content = line[2:-1]  # Remove b' and '
                        if content:
                            try:
                                # Handle escape sequences
                                decoded = content.encode().decode('unicode_escape')
                                byte_array = list(decoded.encode('utf-8'))
                                tokenizer_data[str(i)] = byte_array
                            except:
                                # Last resort: treat as raw string
                                byte_array = list(content.encode('utf-8'))
                                tokenizer_data[str(i)] = byte_array
                        else:
                            tokenizer_data[str(i)] = []
            elif line.startswith('b"') and line.endswith('"'):
                try:
                    # Parse the byte string with double quotes
                    byte_str = ast.literal_eval(line)
                    if isinstance(byte_str, bytes):
                        byte_array = list(byte_str)
                        tokenizer_data[str(i)] = byte_array
                    else:
                        byte_array = list(byte_str.encode('utf-8'))
                        tokenizer_data[str(i)] = byte_array
                except Exception as e:
                    # Manual parsing for double quote format
                    content = line[2:-1]  # Remove b" and "
                    if content:
                        try:
                            decoded = content.encode().decode('unicode_escape')
                            byte_array = list(decoded.encode('utf-8'))
                            tokenizer_data[str(i)] = byte_array
                        except:
                            byte_array = list(content.encode('utf-8'))
                            tokenizer_data[str(i)] = byte_array
                    else:
                        tokenizer_data[str(i)] = []
            elif line.startswith('<|') and line.endswith('|>'):
                # Special tokens like <|bicodec_semantic_*|>
                tokenizer_data[str(i)] = line
                if i < 10 or i % 10000 == 0:
                    print(f"Token {i}: {line} (special token)")
            else:
                # Other formats, keep as string
                tokenizer_data[str(i)] = line
                
        # Progress indicator
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} tokens...")
    
    print(f"\nGenerating formatted tokenizer.json with {len(tokenizer_data)} tokens...")
    
    # Write formatted JSON
    output_path = '../assets/model/tokenizer.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
    
    print(f"Tokenizer saved to {output_path}")
    
    # Verify the result
    print("\nVerification:")
    print(f"Token 0: {tokenizer_data['0']}")
    print(f"Token 1: {tokenizer_data['1']}")
    print(f"Token 40: {tokenizer_data['40']}")
    print(f"Token 256: {tokenizer_data['256']}")
    print(f"Last token: {tokenizer_data[str(len(tokenizer_data)-1)]}")
    
    # Check for some sample tokens
    sample_indices = [1000, 30000, 50000, 70000]
    for idx in sample_indices:
        if str(idx) in tokenizer_data:
            print(f"Token {idx}: {tokenizer_data[str(idx)]}")

if __name__ == '__main__':
    rebuild_tokenizer_formatted()