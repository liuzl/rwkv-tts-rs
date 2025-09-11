import json

# Load and verify the JSON file
with open('rwkv_vocab_v20230424_sparktts_spct_tokens.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total tokens: {len(data)}')
print(f'First key: {list(data.keys())[0]}')
print(f'Last key: {list(data.keys())[-1]}')
print(f'Sample entries:')
for i, (k, v) in enumerate(list(data.items())[:3]):
    print(f'  {k}: {repr(v)}')
print('...')
for i, (k, v) in enumerate(list(data.items())[-3:]):
    print(f'  {k}: {repr(v)}')

# Check if format matches reference
print('\nFormat verification:')
print(f'Keys are numeric strings: {all(k.isdigit() for k in list(data.keys())[:10])}')
print(f'Values are token strings: {all(isinstance(v, str) for v in list(data.values())[:10])}')