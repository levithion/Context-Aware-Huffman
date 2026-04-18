import re

def tokenize(text, mode='word'):
    if mode == 'word':
        return [t for t in re.split(r'(\s+)', text) if t]
    else:
        return list(text)

def detokenize(tokens, mode='word'):
    return "".join(tokens)
