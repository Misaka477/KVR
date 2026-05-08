import pathlib

search_keys = ['modelscope', 'huggingface', '.cache']
for p in pathlib.Path('.').rglob('*.py'):
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
        if any(k in text for k in search_keys):
            print(p)
            for i, line in enumerate(text.splitlines(), 1):
                if any(k in line for k in search_keys):
                    print(f"  L{i}: {line.strip()[:120]}")
    except Exception as e:
        pass