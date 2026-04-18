# app.py
"""
Streamlit UI for the Context-Aware Huffman compressor/decompressor.
Updated to show correct comparison chart: Original vs Regular Huffman vs Context-Aware Huffman.
Run:
    pip install streamlit matplotlib
    streamlit run app.py
"""
from collections import defaultdict, Counter, deque
import heapq, struct, json, io, math, os, time, re
import streamlit as st
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt


# -------------------------
# Huffman utilities (adapted)
# -------------------------
class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_tree(freq_map: Dict[Any, int]):
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


def build_tree_deterministic(freq_map: Dict[Any, int]):
    heap = []
    for sym, f in freq_map.items():
        # use stringified symbol as deterministic tie-breaker
        heap.append((f, str(sym), Node(sym, f)))
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


def get_codes_from_tree(root: Node):
    codes = {}
    if root is None:
        return codes

    def dfs(node, prefix):
        if node.symbol is not None:
            codes[node.symbol] = prefix if prefix != "" else "0"
            return
        if node.left:
            dfs(node.left, prefix + "0")
        if node.right:
            dfs(node.right, prefix + "1")

    dfs(root, "")
    return codes


# tokenizers
def tokenize(text: str, mode="word"):
    if mode == "word":
        return [t for t in re.split(r'(\s+)', text) if t]
    else:
        return list(text)

def detokenize(tokens: List[str], mode="word"):
    return "".join(tokens)

# entropy functions
def calc_shannon_entropy(freq_map: dict):
    total = sum(freq_map.values())
    if total == 0: return 0.0
    return -sum((f/total) * math.log2(f/total) for f in freq_map.values() if f > 0)

def calc_conditional_entropy(context_freqs: dict):
    total_tokens = sum(sum(f.values()) for f in context_freqs.values())
    if total_tokens == 0: return 0.0
    cond_entropy = 0.0
    for ctx, freq_map in context_freqs.items():
        ctx_total = sum(freq_map.values())
        ctx_prob = ctx_total / total_tokens
        ctx_entropy = calc_shannon_entropy(freq_map)
        cond_entropy += ctx_prob * ctx_entropy
    return cond_entropy


# bit I/O
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
            self.write_bit(1 if ch == "1" else 0)

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
        return "".join(bits)


# helpers
ESC_BASE = "<ESC>"


def _make_unique_token(base, existing):
    token = base
    i = 0
    while token in existing:
        i += 1
        token = f"{base}_{i}"
    return token


# -------------------------
# Encoding-size calculators (regular and context-aware)
# -------------------------
def build_global_codebook(tokens: List[str]):
    freq = Counter(tokens)
    root = build_tree_deterministic(freq)
    codes = get_codes_from_tree(root)
    return codes, freq


def build_context_codebooks(tokens: List[str], order=1):
    context_freq = defaultdict(Counter)
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i - order) : i]) if order > 0 else tuple()
        context_freq[ctx][tokens[i]] += 1
    context_codes = {}
    context_freqs = {}
    for ctx, freq_map in context_freq.items():
        root = build_tree_deterministic(freq_map)
        context_codes[ctx] = get_codes_from_tree(root)
        context_freqs[ctx] = freq_map
    return context_codes, context_freqs


def global_huffman_encode_bits(tokens: List[str], global_codes: Dict[str, str]):
    bits = 0
    for t in tokens:
        bits += len(global_codes[t])
    return bits


def context_huffman_encode_bits(
    tokens: List[str],
    context_codes: Dict[tuple, Dict[str, str]],
    global_codes: Dict[str, str],
    order=1,
    esc_token=None,
):
    bits = 0
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i - order) : i]) if order > 0 else tuple()
        codes = context_codes.get(ctx)
        t = tokens[i]
        if codes and t in codes:
            bits += len(codes[t])
        else:
            # escape then global
            if codes and esc_token in codes:
                bits += len(codes[esc_token])
            bits += len(global_codes[t])
    return bits


# -------------------------
# In-memory compress / decompress (fileless)
# -------------------------
def compress_bytes(corpus_list: List[str], mode="word", order=1) -> Tuple[bytes, dict]:
    """
    Returns (bytes_of_file, header_dict) and computes sizes for comparison.
    """
    start_time = time.time()
    joined = "\n".join(corpus_list)
    tokens = tokenize(joined, mode=mode)

    # Build frequency maps
    global_freq = Counter(tokens)
    context_freq = defaultdict(Counter)
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i - order) : i]) if order > 0 else tuple()
        context_freq[ctx][tokens[i]] += 1

    existing = set(global_freq.keys()) | {
        t for ctx in context_freq.values() for t in ctx
    }
    esc_token = _make_unique_token(ESC_BASE, existing)

    # Ensure ESC present in each context's freq map
    for freq_map in context_freq.values():
        if esc_token not in freq_map:
            freq_map[esc_token] += 1

    # Build deterministic trees and codes
    global_root = build_tree_deterministic(global_freq)
    global_codes = get_codes_from_tree(global_root)
    context_codes = {}
    for ctx, freq_map in context_freq.items():
        root = build_tree_deterministic(freq_map)
        context_codes[ctx] = get_codes_from_tree(root)

    # Compute sizes (bits) for regular and context-aware encodings
    global_bits = global_huffman_encode_bits(tokens, global_codes)
    ctx_bits = context_huffman_encode_bits(
        tokens, context_codes, global_codes, order=order, esc_token=esc_token
    )

    # Encode bitstream
    writer = BitWriter()
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i - order) : i]) if order > 0 else tuple()
        codes = context_codes.get(ctx)
        t = tokens[i]
        if codes and t in codes:
            writer.write_bits(codes[t])
        else:
            if codes and esc_token in codes:
                writer.write_bits(codes[esc_token])
            writer.write_bits(global_codes[t])
    bit_bytes = writer.get_bytes()

    global_entropy = calc_shannon_entropy(global_freq)
    conditional_entropy = calc_conditional_entropy(context_freq)
    encoding_time = time.time() - start_time

    header = {
        "version": 1,
        "mode": mode,
        "order": order,
        "total_tokens": len(tokens),
        "esc_token": esc_token,
        "global_freq": dict(global_freq),
        "contexts": [
            {"ctx": list(ctx), "freq": dict(freq_map)}
            for ctx, freq_map in context_freq.items()
        ],
        # include size metrics for convenience
        "metrics": {
            "global_bits": global_bits,
            "ctx_bits": ctx_bits,
            "global_bytes": math.ceil(global_bits / 8),
            "ctx_bytes": math.ceil(ctx_bits / 8),
            "original_bytes": len(joined.encode("utf-8")),
            "global_entropy": global_entropy,
            "conditional_entropy": conditional_entropy,
            "encoding_time": encoding_time,
        },
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    out = io.BytesIO()
    out.write(struct.pack(">I", len(header_bytes)))
    out.write(header_bytes)
    out.write(bit_bytes)
    return out.getvalue(), header


def decompress_bytes(data: bytes) -> Tuple[List[str], str, dict]:
    buf = io.BytesIO(data)
    header_len_bytes = buf.read(4)
    if len(header_len_bytes) < 4:
        raise ValueError("File too short or missing header length")
    header_len = struct.unpack(">I", header_len_bytes)[0]
    header_bytes = buf.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))

    mode = header["mode"]
    order = header["order"]
    total_tokens = header["total_tokens"]
    esc_token = header["esc_token"]

    global_freq = Counter(header["global_freq"])
    global_root = build_tree_deterministic(global_freq)

    context_trees = {}
    for entry in header["contexts"]:
        ctx = tuple(entry["ctx"])
        freq_map = Counter(entry["freq"])
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
            nodeg = global_root
            while nodeg.symbol is None:
                b = reader.read_bit()
                nodeg = nodeg.left if b == 0 else nodeg.right
            gs = nodeg.symbol
            out_tokens.append(gs)
            prev.append(gs)

    text = detokenize(out_tokens, mode=mode)
    return out_tokens, text, header


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Context-Aware Huffman Compressor", layout="wide")
st.title("🔧 Context-Aware Huffman Compressor / Decompressor")

st.markdown(
    """
    Upload text files or paste text, choose tokenizer mode and context order, compress to a single `.bin` file,
    and then decompress to verify. The app now displays a correct comparison chart showing:
    Original bytes vs Regular Huffman bytes vs Context-Aware Huffman bytes.
    """
)

tab1, tab2, tab3 = st.tabs(["Compress", "Decompress", "Sample experiments"])

# session storage
if "last_compressed" not in st.session_state:
    st.session_state["last_compressed"] = None
if "last_header" not in st.session_state:
    st.session_state["last_header"] = None
if "last_original_text" not in st.session_state:
    st.session_state["last_original_text"] = None


# -------------------------------------
# Visualization Utilities (NEW GRAPHS)
# -------------------------------------
def show_comparison_chart(
    original_bytes, global_bytes, ctx_bytes, title="Compression sizes"
):
    labels = ["Original", "Regular Huffman", "Context-Aware Huffman"]
    values = [original_bytes, global_bytes, ctx_bytes]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["gray", "skyblue", "lightgreen"])
    ax.set_title(title)
    ax.set_ylabel("Bytes")
    maxv = max(values) if values else 1
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + maxv * 0.01, str(v), ha="center")
    st.pyplot(fig)


def show_efficiency_chart(original_bytes, global_bytes, ctx_bytes):
    efficiencies = [
        (1 - global_bytes / original_bytes) * 100,
        (1 - ctx_bytes / original_bytes) * 100,
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Regular Huffman", "Context-Aware Huffman"],
        efficiencies,
        color=["skyblue", "lightgreen"],
    )
    ax.set_title("Compression Efficiency (%)")
    ax.set_ylabel("Reduction (%)")
    for bar, v in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.2f}%", ha="center")
    st.pyplot(fig)


def show_bit_distribution_chart(global_freq, contexts_list):
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        # Convert contexts list back to dictionary format
        context_freqs = {}
        for entry in contexts_list:
            ctx = tuple(entry["ctx"])
            freq_dict = entry["freq"]
            context_freqs[ctx] = freq_dict

        # Estimate code lengths from frequencies
        total_global = sum(global_freq.values()) or 1
        global_lengths = [
            int(-np.log2(freq / total_global)) if freq > 0 else 0
            for freq in global_freq.values()
        ]

        ctx_lengths = []
        for ctx, freq_dict in context_freqs.items():
            total_ctx = sum(freq_dict.values()) or 1
            for freq in freq_dict.values():
                if freq > 0:
                    ctx_lengths.append(int(-np.log2(freq / total_ctx)))

        if not global_lengths or not ctx_lengths:
            st.warning("Insufficient frequency data to plot bit distribution.")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            global_lengths,
            bins=range(1, max(global_lengths) + 2),
            alpha=0.6,
            label="Regular Huffman (estimated)",
            color="skyblue",
        )
        ax.hist(
            ctx_lengths,
            bins=range(1, max(ctx_lengths) + 2),
            alpha=0.6,
            label="Context-Aware Huffman (estimated)",
            color="lightgreen",
        )
        ax.set_xlabel("Estimated Code Length (bits)")
        ax.set_ylabel("Symbol Count")
        ax.set_title("Bit Distribution per Token (Estimated)")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting bit distribution: {e}")


def show_freq_vs_code_length(global_freq, global_codes):
    freq_values = []
    code_lengths = []
    for token, freq in global_freq.items():
        if token in global_codes:
            freq_values.append(freq)
            code_lengths.append(len(global_codes[token]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(freq_values, code_lengths, alpha=0.7, color="purple")
    ax.set_xscale("log")
    ax.set_xlabel("Token Frequency (log scale)")
    ax.set_ylabel("Code Length (bits)")
    ax.set_title("Token Frequency vs Code Length (Regular Huffman)")
    st.pyplot(fig)


def show_theoretical_entropy_chart(global_entropy, global_actual, cond_entropy, ctx_actual):
    labels = ["Regular Huffman", "Context-Aware"]
    entropies = [global_entropy, cond_entropy]
    actuals = [global_actual, ctx_actual]
    
    x = [0, 1]
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar([i - width/2 for i in x], entropies, width, label='Theoretical Entropy Limit', color="gray", alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in x], actuals, width, label='Actual Huffman Coding', color="skyblue")

    ax.set_ylabel('Bits per token')
    ax.set_title('Theoretical Minimum vs Actual Achieved Bits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + max(max(entropies), max(actuals))*0.02, f'{yval:.2f}', ha='center', fontsize=9)
    
    st.pyplot(fig)

with tab1:
    st.header("Compress text -> .bin")
    uploaded_files = st.file_uploader(
        "Upload one or more text files (.txt). You can also paste text below.",
        accept_multiple_files=True,
        type=["txt"],
    )
    paste = st.text_area(
        "Or paste text here (ignored if you uploaded files)", height=150
    )
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox(
            "Tokenization mode",
            options=["word", "char"],
            index=0,
            help="word: whitespace tokens. char: character tokens.",
        )
        order = st.number_input(
            "Context order (previous tokens to use as context)",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
        )
    with col2:
        out_name = st.text_input(
            "Output filename (downloaded)", value="compressed_sample.bin"
        )
        do_compress = st.button("Compress")

    if do_compress:
        # prepare corpus list
        if uploaded_files:
            corpus = []
            for f in uploaded_files:
                try:
                    raw = f.read()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    corpus.append(raw)
                except Exception as e:
                    st.error(f"Could not read file {f.name}: {e}")
            if not corpus:
                st.error("No readable files uploaded.")
                st.stop()
        elif paste.strip():
            corpus = [paste]
        else:
            st.error("Please upload files or paste text to compress.")
            st.stop()

        try:
            data_bytes, header = compress_bytes(corpus, mode=mode, order=order)
            st.session_state["last_compressed"] = data_bytes
            st.session_state["last_header"] = header
            st.session_state["last_original_text"] = "\n".join(corpus)
            st.success("Compression complete.")
            st.write("Header (decoded):")
            st.json(header)
            compressed_size = len(data_bytes)
            st.write(f"Compressed file size (on-disk): {compressed_size} bytes.")
            original_size = header["metrics"]["original_bytes"]
            global_bytes = header["metrics"]["global_bytes"]
            ctx_bytes = header["metrics"]["ctx_bytes"]
            encoding_time = header["metrics"].get("encoding_time", 0.001)

            st.write(f"Original text size (bytes): {original_size}")
            st.write(f"Regular Huffman (bytes): {global_bytes}")
            st.write(f"Context-Aware Huffman (bytes): {ctx_bytes}")
            
            st.write(f"**Encoding Speed:** {original_size / 1024 / 1024 / encoding_time if encoding_time > 0 else 0:.2f} MB/s")

            try:
                ratio = (
                    original_size / compressed_size
                    if compressed_size > 0
                    else float("inf")
                )
                st.write(f"Compression ratio (original / compressed): {ratio:.3f}")
            except Exception:
                pass

            # show correct comparison chart
            show_comparison_chart(
                original_size,
                global_bytes,
                ctx_bytes,
                title="Compression sizes (Original vs Regular vs Context-Aware)",
            )
            
            # --- Entropy Metrics ---
            act_global = header["metrics"]["global_bits"] / header["total_tokens"]
            act_ctx = header["metrics"]["ctx_bits"] / header["total_tokens"]
            show_theoretical_entropy_chart(
                header["metrics"]["global_entropy"],
                act_global,
                header["metrics"]["conditional_entropy"],
                act_ctx
            )

            # --- New analytical charts ---
            # download button
            st.download_button(
                label="Download compressed file",
                data=data_bytes,
                file_name=out_name,
                mime="application/octet-stream",
            )
            
            show_efficiency_chart(original_size, global_bytes, ctx_bytes)
            show_bit_distribution_chart(header["global_freq"], header["contexts"])
            show_freq_vs_code_length(
                header["global_freq"],
                get_codes_from_tree(
                    build_tree_deterministic(Counter(header["global_freq"]))
                ),
            )

        except Exception as e:
            st.exception(e)

with tab2:
    st.header("Decompress .bin -> text")
    uploaded_bin = st.file_uploader(
        "Upload compressed .bin file", type=None, accept_multiple_files=False
    )
    use_last = st.checkbox(
        "Or use the last compression result performed in this session", value=True
    )
    do_decompress = st.button("Decompress")
    if do_decompress:
        data = None
        if use_last and st.session_state["last_compressed"] is not None:
            data = st.session_state["last_compressed"]
            header = st.session_state.get("last_header")
        elif uploaded_bin is not None:
            try:
                data = uploaded_bin.read()
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
                st.stop()
        else:
            st.error(
                "No compressed data available. Upload a .bin or use last in-session result."
            )
            st.stop()

        try:
            toks, text, header = decompress_bytes(data)
            st.success("Decompression successful.")
            st.write("Header (decoded):")
            st.json(header)
            st.write(
                f"Decompressed tokens: {len(toks)} (header expected {header.get('total_tokens')})"
            )
            st.download_button(
                "Download decompressed text",
                data=text.encode("utf-8"),
                file_name="decompressed.txt",
                mime="text/plain",
            )
            st.subheader("Preview (first 2000 characters):")
            st.code(text[:2000])
            # verification if we have original in session
            original = st.session_state.get("last_original_text")
            if original:
                tok_orig = tokenize(original, mode=header.get("mode", "word"))
                ok = tok_orig == toks
                if ok:
                    st.success(
                        "Verification: decompressed token sequence matches the last original text in this session."
                    )
                else:
                    st.warning(
                        "Verification: decompressed token sequence DOES NOT match last original text in this session."
                    )
                    for i, (a, b) in enumerate(zip(tok_orig, toks)):
                        if a != b:
                            st.write(
                                f"First mismatch at token {i}: original='{a}' vs decompressed='{b}'"
                            )
                            break
        except Exception as e:
            st.exception(e)

with tab3:
    st.header("Sample experiments (quick run with synthetic corpora)")
    st.write(
        "Generate simple natural-language and code-like corpora and run the compression experiment."
    )
    if st.button("Run sample experiment"):
        # create sample corpora (same idea as user's make_strong_context_corpus)
        s1 = ("the quick brown fox jumps over the lazy dog " * 150).strip()
        s2 = ("the the the the the the the " * 150).strip()
        s3 = ("in the beginning in the beginning in the beginning " * 120).strip()
        nl = [s1, s2, s3]

        c1 = ("for i in range(10):\n    print(i)\n" * 150).strip()
        c2 = ("def foo():\n    pass\n" * 120).strip()
        c3 = ("if x > 0:\n    x -= 1\nelse:\n    x += 1\n" * 100).strip()
        code = [c1, c2, c3]

        sample = nl + ["\n--- CODE CORPUS BELOW ---\n"] + code
        st.write("Compressing sample ...")
        data_bytes, header = compress_bytes(sample, mode="word", order=1)
        st.write("Header summary:")
        st.json(
            {
                k: header[k]
                for k in ("version", "mode", "order", "total_tokens", "esc_token")
            }
        )
        st.write(f"Compressed sample size: {len(data_bytes)} bytes")
        st.download_button(
            "Download sample compressed file", data=data_bytes, file_name="sample.bin"
        )


st.markdown("---")
st.caption(
    "Notes: header contains frequency maps and size metrics used to compute and display the comparison chart."
)