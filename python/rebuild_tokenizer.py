import json

# 读取vocab.txt文件
with open('vocab.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 创建tokenizer字典
tokenizer_dict = {}

for i, line in enumerate(lines):
    token = line.strip()
    
    if i == 0:
        # 第一个token是特殊token，保持字符串格式
        tokenizer_dict[str(i)] = token
    else:
        # 其他token转换为字节数组
        if token.startswith("b'") and token.endswith("'"):
            # 处理b'xxx'格式的字节字符串
            try:
                # 移除b'和'，然后解码
                byte_str = token[2:-1]
                # 处理转义字符
                byte_str = byte_str.encode().decode('unicode_escape')
                byte_values = [ord(c) for c in byte_str]
                tokenizer_dict[str(i)] = byte_values
            except Exception as e:
                print(f"处理token {i} 时出错: {token}, 错误: {e}")
                # 如果解析失败，使用UTF-8编码
                byte_values = token.encode('utf-8')
                tokenizer_dict[str(i)] = list(byte_values)
        elif token.startswith('<|') and token.endswith('|>'):
            # 特殊token保持字符串格式
            tokenizer_dict[str(i)] = token
        else:
            # 普通文本转换为UTF-8字节数组
            byte_values = token.encode('utf-8')
            tokenizer_dict[str(i)] = list(byte_values)

# 写入JSON文件（单行格式）
with open('../assets/model/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_dict, f, separators=(',', ':'), ensure_ascii=False)

print(f'生成了包含 {len(tokenizer_dict)} 个token的词表')
print(f'第0个token: {tokenizer_dict["0"]}')
print(f'第1个token: {tokenizer_dict["1"]}')
print(f'第256个token: {tokenizer_dict["256"]}')
print(f'最后一个token: {tokenizer_dict[str(len(tokenizer_dict)-1)]}')

# 验证一些中文token
for i in range(min(100, len(tokenizer_dict))):
    token_id = str(i)
    if token_id in tokenizer_dict:
        token_value = tokenizer_dict[token_id]
        if isinstance(token_value, list) and len(token_value) > 0:
            # 尝试解码为UTF-8看是否包含中文
            try:
                decoded = bytes(token_value).decode('utf-8')
                if any('\u4e00' <= char <= '\u9fff' for char in decoded):
                    print(f'发现中文token {i}: {decoded}')
            except:
                pass