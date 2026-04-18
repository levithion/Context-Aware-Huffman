import heapq
from collections import defaultdict, Counter, deque
import math
import json
import os
import struct
import io
import time
import argparse
import sys


def calc_shannon_entropy(freq_map):
    total = sum(freq_map.values())
    if total == 0: return 0.0
    return -sum((f/total) * math.log2(f/total) for f in freq_map.values() if f > 0)

def calc_conditional_entropy(context_freqs):
    total_tokens = sum(sum(f.values()) for f in context_freqs.values())
    if total_tokens == 0: return 0.0
    cond_entropy = 0.0
    for ctx, freq_map in context_freqs.items():
        ctx_total = sum(freq_map.values())
        ctx_prob = ctx_total / total_tokens
        ctx_entropy = calc_shannon_entropy(freq_map)
        cond_entropy += ctx_prob * ctx_entropy
    return cond_entropy

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(freq_map):
    heap = [Node(sym, f) for sym, f in freq_map.items()]
    if not heap:
        return None
    heapq.heapify(heap)
    if len(heap) == 1:
        single = heapq.heappop(heap)
        return Node(None, single.freq, single, None)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = Node(None, a.freq + b.freq, a, b)
        heapq.heappush(heap, merged)
    return heap[0]

def build_tree_deterministic(freq_map):
    """
    Build Huffman tree deterministically using (freq, symbol) as tie-breaker so
    the decoder can reconstruct the same tree from the same frequency map.
    """
    heap = []
    for sym, f in freq_map.items():
        heap.append((f, sym, Node(sym, f)))
    if not heap:
        return None
    heapq.heapify(heap)
    if len(heap) == 1:
        f, sym, single = heapq.heappop(heap)
        return Node(None, single.freq, single, None)
    while len(heap) > 1:
        f1, s1, a = heapq.heappop(heap)
        f2, s2, b = heapq.heappop(heap)
        merged = Node(None, a.freq + b.freq, a, b)
        tie = s1 if s1 <= s2 else s2
        heapq.heappush(heap, (merged.freq, tie, merged))
    return heap[0][2]


def get_codes_from_tree(root):
    codes = {}
    if root is None:
        return codes
    def dfs(node, prefix):
        if node.symbol is not None:
            codes[node.symbol] = prefix if prefix != "" else "0"
            return
        if node.left: dfs(node.left, prefix + "0")
        if node.right: dfs(node.right, prefix + "1")
    dfs(root, "")
    return codes

import re

def tokenize(text, mode='word'):
    if mode == 'word':
        # Split by contiguous whitespace, preserving the whitespaces as tokens
        return [t for t in re.split(r'(\s+)', text) if t]
    else:
        return list(text)

def detokenize(tokens, mode='word'):
    # Rejoin tokens perfectly since whitespace is preserved in tokens list
    return "".join(tokens)

def build_global_codebook(tokens):
    freq = Counter(tokens)
    root = build_tree(freq)
    return get_codes_from_tree(root), freq

def build_context_codebooks(tokens, order=1):
    context_freq = defaultdict(Counter)
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i-order):i]) if order>0 else tuple()
        context_freq[ctx][tokens[i]] += 1
    context_codes = {}
    context_freqs = {}
    for ctx, freq_map in context_freq.items():
        root = build_tree(freq_map)
        codes = get_codes_from_tree(root)
        context_codes[ctx] = codes
        context_freqs[ctx] = freq_map
    return context_codes, context_freqs

def context_huffman_encode(tokens, context_codes, global_codes, order=1):
    bits = 0
    encoded_bits = []
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i-order):i]) if order>0 else tuple()
        codes = context_codes.get(ctx)
        if not codes or tokens[i] not in codes:
            code = global_codes.get(tokens[i])
            if code is None:
                code = format(hash(tokens[i]) & 0xffff, '016b')
            encoded_bits.append(code)
            bits += len(code)
        else:
            code = codes[tokens[i]]
            encoded_bits.append(code)
            bits += len(code)
    return bits, "".join(encoded_bits)

def global_huffman_encode(tokens, global_codes):
    bits = 0
    for t in tokens:
        bits += len(global_codes[t])
    return bits


def make_strong_context_corpus():
    s1 = ("the quick brown fox jumps over the lazy dog " * 200).strip()
    s2 = ("the the the the the the the " * 200).strip()
    s3 = ("in the beginning in the beginning in the beginning " * 150).strip()
    nl = [s1, s2, s3]

    c1 = ("for i in range(10):\n    print(i)\n" * 200).strip()
    c2 = ("def foo():\n    pass\n" * 150).strip()
    c3 = ("if x > 0:\n    x -= 1\nelse:\n    x += 1\n" * 120).strip()
    code = [c1, c2, c3]
    return nl, code


class BitWriter:
    def __init__(self):
        self.bytes = bytearray()
        self.current = 0
        self.nbits = 0

    def write_bit(self, b: int):
        self.current = (self.current << 1) | (1 if b else 0)
        self.nbits += 1
        if self.nbits == 8:
            self.bytes.append(self.current)
            self.current = 0
            self.nbits = 0

    def write_bits(self, bitstring: str):
        for ch in bitstring:
            self.write_bit(1 if ch == '1' else 0)

    def get_bytes(self) -> bytes:
        if self.nbits > 0:
            self.current = self.current << (8 - self.nbits)
            self.bytes.append(self.current)
            self.current = 0
            self.nbits = 0
        return bytes(self.bytes)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.current = 0
        self.nbits = 0

    def read_bit(self):
        if self.nbits == 0:
            if self.pos >= len(self.data):
                raise EOFError("No more bits available")
            self.current = self.data[self.pos]
            self.pos += 1
            self.nbits = 8
        bit = (self.current >> (self.nbits - 1)) & 1
        self.nbits -= 1
        return bit

    def read_bits(self, n):
        bits = []
        for _ in range(n):
            bits.append(str(self.read_bit()))
        return ''.join(bits)


ESC_BASE = "<ESC>"

def _make_unique_token(base, existing):
    token = base
    i = 0
    while token in existing:
        i += 1
        token = f"{base}_{i}"
    return token

def _needs_esc(ctx_vocab: set, global_vocab: set) -> bool:
    """
    FIX: A context only needs an ESC symbol in its tree when its observed
    vocabulary is a strict subset of the global vocabulary.  Adding ESC
    unconditionally to every context (even those that cover all tokens)
    inflates every tree and wastes bits on a code path that is never taken,
    shrinking the effective compression ratio.
    """
    return ctx_vocab < global_vocab          # strict subset → may see unseen tokens

def compress_to_file(filepath, corpus_list, mode='word', order=1):
    """Compress the corpus_list to a file in a self-contained format.

    Format: 4-byte big-endian header length, header JSON, then raw bitstream bytes.
    Header fields: version, mode, order, total_tokens, esc_token, global_freq, contexts
    contexts: list of {"ctx": [...], "freq": {token: count}, "has_esc": bool}
    """
    joined = "\n".join(corpus_list)
    tokens = tokenize(joined, mode=mode)

    global_freq = Counter(tokens)
    global_vocab = set(global_freq.keys())

    context_freq = defaultdict(Counter)
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i-order):i]) if order>0 else tuple()
        context_freq[ctx][tokens[i]] += 1

    existing = set(global_freq.keys()) | {t for ctx in context_freq.values() for t in ctx}
    esc_token = _make_unique_token(ESC_BASE, existing)

    # FIX: only inject ESC where the context vocabulary is a strict subset of
    # the global vocabulary (i.e. some tokens might never have been seen after
    # this context and would need the fallback path).
    ctx_has_esc = {}
    for ctx, freq_map in context_freq.items():
        ctx_vocab = set(freq_map.keys())
        if _needs_esc(ctx_vocab, global_vocab):
            freq_map[esc_token] += 1
            ctx_has_esc[ctx] = True
        else:
            ctx_has_esc[ctx] = False

    global_root = build_tree_deterministic(global_freq)
    global_codes = get_codes_from_tree(global_root)
    context_codes = {}
    for ctx, freq_map in context_freq.items():
        root = build_tree_deterministic(freq_map)
        context_codes[ctx] = get_codes_from_tree(root)

    writer = BitWriter()
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i-order):i]) if order>0 else tuple()
        codes = context_codes.get(ctx)
        t = tokens[i]
        if codes and t in codes:
            writer.write_bits(codes[t])
        else:
            # ESC fallback: emit ESC from context tree then global code
            if codes and esc_token in codes:
                writer.write_bits(codes[esc_token])
            writer.write_bits(global_codes[t])

    bit_bytes = writer.get_bytes()

    header = {
        'version': 2,
        'mode': mode,
        'order': order,
        'total_tokens': len(tokens),
        'esc_token': esc_token,
        'global_freq': dict(global_freq),
        'contexts': [
            {
                'ctx': list(ctx),
                'freq': dict(freq_map),
                'has_esc': ctx_has_esc[ctx]
            }
            for ctx, freq_map in context_freq.items()
        ]
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode('utf-8')
    with open(filepath, 'wb') as f:
        f.write(struct.pack('>I', len(header_bytes)))
        f.write(header_bytes)
        f.write(bit_bytes)


def decompress_file(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    buf = io.BytesIO(raw)
    header_len_bytes = buf.read(4)
    if len(header_len_bytes) < 4:
        raise ValueError("File too short or missing header length")
    header_len = struct.unpack('>I', header_len_bytes)[0]
    header_bytes = buf.read(header_len)
    header = json.loads(header_bytes.decode('utf-8'))

    mode = header['mode']
    order = header['order']
    total_tokens = header['total_tokens']
    esc_token = header['esc_token']

    global_freq = Counter(header['global_freq'])
    global_root = build_tree_deterministic(global_freq)

    context_trees = {}
    for entry in header['contexts']:
        ctx = tuple(entry['ctx'])
        freq_map = Counter(entry['freq'])
        context_trees[ctx] = build_tree_deterministic(freq_map)

    bit_bytes = buf.read()
    reader = BitReader(bit_bytes)

    prev = deque(maxlen=order)
    out_tokens = []
    for _ in range(total_tokens):
        ctx = tuple(prev)
        tree = context_trees.get(ctx)
        if tree is None:
            node = global_root
            while node.symbol is None:
                b = reader.read_bit()
                node = node.left if b == 0 else node.right
            symbol = node.symbol
            out_tokens.append(symbol)
            prev.append(symbol)
            continue

        node = tree
        while node.symbol is None:
            b = reader.read_bit()
            node = node.left if b == 0 else node.right
        symbol = node.symbol
        if symbol != esc_token:
            out_tokens.append(symbol)
            prev.append(symbol)
        else:
            node = global_root
            while node.symbol is None:
                b = reader.read_bit()
                node = node.left if b == 0 else node.right
            gs = node.symbol
            out_tokens.append(gs)
            prev.append(gs)

    text = detokenize(out_tokens, mode=mode)
    return out_tokens, text


def run_experiment(corpus_list, mode='word', order=1, label="Dataset"):
    start_time = time.time()
    joined = "\n".join(corpus_list)
    tokens = tokenize(joined, mode=mode)

    global_codes, global_freq = build_global_codebook(tokens)
    context_codes, context_freqs = build_context_codebooks(tokens, order=order)

    global_bits = global_huffman_encode(tokens, global_codes)
    ctx_bits, _ = context_huffman_encode(tokens, context_codes, global_codes, order=order)
    
    encoding_time = time.time() - start_time
    
    # Calculate Shannon Entropy vs Current Huffman Efficiency
    global_entropy = calc_shannon_entropy(global_freq)
    conditional_entropy = calc_conditional_entropy(context_freqs)

    original_bytes = len(joined.encode('utf-8'))
    global_bytes = math.ceil(global_bits / 8)
    ctx_bytes = math.ceil(ctx_bits / 8)
    ratio_global = original_bytes / global_bytes if global_bytes>0 else float('inf')
    ratio_ctx = original_bytes / ctx_bytes if ctx_bytes>0 else float('inf')

    print(f"\n--- {label} (mode={mode}, context_order={order}) ---")
    print(f"Original Size (bytes): {original_bytes} bytes")
    print(f"Regular Huffman Size (bytes): {global_bytes} bytes")
    print(f"Context-Aware Huffman Size (bytes): {ctx_bytes} bytes")
    print(f"Regular Compression Ratio: {ratio_global:.3f}")
    print(f"Context-Aware Compression Ratio: {ratio_ctx:.3f}")
    print(f"Encoding Speed: {original_bytes / 1024 / 1024 / encoding_time if encoding_time > 0 else 0:.2f} MB/s")
    
    print(f"\nTheoretical Limits (Bits per token):")
    print(f"  Global Shannon Entropy:      {global_entropy:.3f} bits/token")
    print(f"  Actual Regular Huffman:      {global_bits/len(tokens):.3f} bits/token")
    print(f"  Conditional Shannon Entropy: {conditional_entropy:.3f} bits/token")
    print(f"  Actual Context-Aware:        {ctx_bits/len(tokens):.3f} bits/token")

    print("\nTop contexts (by token count) and avg bits/token (global vs context):")
    ctx_stats = []
    for ctx, freq_map in context_freqs.items():
        total = sum(freq_map.values())
        if total < 10:
            continue
        codes = context_codes[ctx]
        avg_ctx_bits = sum((len(codes[t]) * f) for t, f in freq_map.items()) / total
        avg_glob_bits = sum((len(global_codes[t]) * f) for t, f in freq_map.items()) / total
        ctx_stats.append((total, ctx, avg_glob_bits, avg_ctx_bits))
    ctx_stats.sort(reverse=True)
    for total, ctx, ag, ac in ctx_stats[:10]:
        ctx_display = " ".join(ctx) if ctx else "<START>"
        print(f"  ctx='{ctx_display}': count={total}, global_avg_bits={ag:.2f}, context_avg_bits={ac:.2f}")

    return {
        "original_bytes": original_bytes,
        "global_bytes": global_bytes,
        "ctx_bytes": ctx_bytes,
        "ratio_global": ratio_global,
        "ratio_ctx": ratio_ctx
    }

def run_sample_experiment():
    print("Running experiments and printing compression/decompression proofs for each corpus...\n")
    nl_corpus, code_corpus = make_strong_context_corpus()

    res_nl = run_experiment(nl_corpus, mode='word', order=1, label="Natural Language")
    res_code = run_experiment(code_corpus, mode='word', order=1, label="Programming Code")

    sample = nl_corpus + ["\n--- CODE CORPUS BELOW ---\n"] + code_corpus
    sample_out = 'sample_file.bin'
    sample_original_txt = 'sample_file_original.txt'

    print(f"\n--- Proof for: Sample file (Natural Language + Programming Code) ---")
    joined_original = "\n".join(sample)
    with open(sample_original_txt, 'w', encoding='utf-8') as fo:
        fo.write(joined_original)
    print(f"Original sample file written to: {sample_original_txt} ({len(joined_original.encode('utf-8'))} bytes)")

    print(f"Compressing sample file to {sample_out}...")
    combined_res = run_experiment(sample, mode='word', order=1, label="Sample File")
    compress_to_file(sample_out, sample, mode='word', order=1)

    with open(sample_out, 'rb') as f:
        header_len_bytes = f.read(4)
        header_len = struct.unpack('>I', header_len_bytes)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode('utf-8'))
        bitstream = f.read()

    print("Header JSON (decoded):")
    print(json.dumps(header, indent=2, ensure_ascii=False))
    total_size = os.path.getsize(sample_out)
    print(f"File size: {total_size} bytes (header {header_len} bytes, bitstream {len(bitstream)} bytes)")

    toks_before = tokenize(joined_original, mode='word')
    toks_after, text_after = decompress_file(sample_out)
    if toks_before == toks_after:
        print(f"Verification PASSED: decompressed token sequence matches original ({len(toks_after)} tokens).")
    else:
        print(f"Verification FAILED: original {len(toks_before)} tokens, decompressed {len(toks_after)} tokens.")
        for i, (a, b) in enumerate(zip(toks_before, toks_after)):
            if a != b:
                print(f"  first mismatch at token {i}: before='{a}' after='{b}'")
                break

    print("First 800 characters of decompressed sample (for spot-check):\n")
    print(text_after[:800])
    sep = "\n--- CODE CORPUS BELOW ---\n"
    pos = text_after.find(sep)
    if pos != -1:
        start = pos + len(sep)
        print("\nFound code separator in decompressed text; showing 800 characters starting at code corpus:\n")
        print(text_after[start:start+800])

    print("\nSample file and original text written.")

    try:
        import matplotlib.pyplot as plt
        labels = ["Original", "Regular Huffman", "Context-Aware Huffman"]
        values = [combined_res['original_bytes'], combined_res['global_bytes'], combined_res['ctx_bytes']]
        plt.figure(figsize=(7,5))
        bars = plt.bar(labels, values, color=['#4C72B0', '#55A868', '#C44E52'])
        plt.title('Compression sizes for sample file')
        plt.ylabel('Bytes')
        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, v + max(values)*0.01, str(v), ha='center')
        out_png = 'compression_comparison.png'
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Saved comparison chart to {out_png}")
    except Exception:
        print("matplotlib not available — install with: python3 -m pip install matplotlib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-Aware Huffman Compressor/Decompressor")
    parser.add_argument("--compress", type=str, help="File to compress")
    parser.add_argument("--decompress", type=str, help="File to decompress")
    parser.add_argument("--out", type=str, help="Output file path")
    parser.add_argument("--mode", type=str, choices=['word', 'char'], default='word', help="Tokenization mode")
    parser.add_argument("--order", type=int, default=1, help="Context order (e.g. 1 to look at 1 previous token)")
    parser.add_argument("--experiment", action='store_true', help="Run the built-in test experiment")
    
    args = parser.parse_args()
    
    if args.experiment or (not args.compress and not args.decompress):
        run_sample_experiment()
    elif args.compress:
        if not args.out:
            print("Please specify an output file with --out")
            sys.exit(1)
        with open(args.compress, 'r', encoding='utf-8') as f:
            corpus = [f.read()]
        
        print(f"Compressing {args.compress} (order={args.order}, mode={args.mode})...")
        start = time.time()
        compress_to_file(args.out, corpus, mode=args.mode, order=args.order)
        print(f"Done in {time.time() - start:.2f} seconds!")
        
    elif args.decompress:
        if not args.out:
            print("Please specify an output file with --out")
            sys.exit(1)
        print(f"Decompressing {args.decompress}...")
        start = time.time()
        tokens, text = decompress_file(args.decompress)
        print(f"Done in {time.time() - start:.2f} seconds!")
        
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Decompressed text written to {args.out}")