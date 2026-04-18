# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **Context-Aware Huffman Coding** compressor/decompressor that improves on standard Huffman coding by building per-context Huffman trees. Instead of one global frequency table, it tracks which tokens follow each context (previous N tokens) and builds separate Huffman trees per context, achieving better compression when token distributions vary by context.

## Running the Project

```bash
# Install dependencies (into venv at .venv/)
pip install streamlit matplotlib numpy

# Run the Streamlit web UI
streamlit run app.py

# Run CLI experiments (default: built-in sample experiment)
python main.py --experiment

# Compress a file
python main.py --compress input.txt --out output.bin --mode word --order 1

# Decompress a file
python main.py --decompress output.bin --out restored.txt

# CLI options: --mode {word,char}, --order N (context window size, default 1)
```

## Architecture

### Two Entry Points, Duplicated Core

**`main.py`** — CLI tool with `argparse`. Contains the canonical compression/decompression logic using file I/O (`compress_to_file`, `decompress_file`), plus `run_experiment` and `run_sample_experiment` for benchmarking.

**`app.py`** — Streamlit web UI. **Duplicates** the core Huffman logic from `main.py` (does not import it). Uses in-memory byte I/O variants (`compress_bytes`, `decompress_bytes`) instead of file paths. Has three tabs: Compress, Decompress, Sample Experiments. Also contains visualization functions (`show_comparison_chart`, `show_efficiency_chart`, `show_bit_distribution_chart`, `show_freq_vs_code_length`, `show_theoretical_entropy_chart`).

> **Important:** Changes to compression logic must be mirrored in both files. The two implementations have minor divergences (e.g., `main.py` v2 uses `_needs_esc` to selectively add ESC tokens; `app.py` v1 adds ESC unconditionally to every context).

### Compression Pipeline

1. **Tokenize** — `tokenize(text, mode)`: `word` mode splits on whitespace boundaries preserving whitespace tokens; `char` mode splits into individual characters.
2. **Build frequency maps** — Global `Counter` over all tokens + per-context `Counter` (context = tuple of previous `order` tokens).
3. **ESC token injection** — A unique `<ESC>` sentinel is added to context frequency maps to handle unseen tokens (fallback to global tree).
4. **Build Huffman trees** — `build_tree_deterministic` (deterministic tie-breaking by symbol) for both global and per-context trees. Codes extracted via `get_codes_from_tree`.
5. **Encode bitstream** — For each token: look up context tree first; if token found, emit context code; otherwise emit ESC code + global code.
6. **Serialize** — Binary format: `[4-byte big-endian header length][JSON header][raw bitstream bytes]`. Header contains version, mode, order, total_tokens, esc_token, global_freq, and all context frequency maps.

### Key Design Decisions

- **Deterministic tree building** (`build_tree_deterministic`): Sorts by `(freq, symbol)` to ensure identical trees across runs, critical for decompression correctness.
- **ESC fallback mechanism**: When a context tree doesn't contain a token, the encoder emits the ESC codeword from the context tree followed by the token's global codeword. `main.py` v2 optimizes this by only injecting ESC where `ctx_vocab ⊊ global_vocab`.
- **Header stores frequency maps, not code tables**: Trees are rebuilt from frequencies during decompression, avoiding code-table serialization format issues.

### Supporting Files

- **`patch.py`** — Standalone `tokenize`/`detokenize` functions (word/char modes).
- **`patch_main.py`** — Imports `main` and re-exposes entropy calculation functions.
- **`paper.tex` / `Latex Code.txt`** — LaTeX source for the academic paper/report.
- **`sample_file.bin`** / **`sample_file_original.txt`** — Pre-generated sample compressed output and its original text.
- **`out.txt`** — Captured output from a sample experiment run.

### Entropy Metrics

The codebase computes and compares:
- **Shannon entropy** (global): theoretical lower bound for memoryless coding
- **Conditional entropy**: theoretical lower bound for context-aware coding
- **Actual bits/token** for both regular and context-aware Huffman

These are displayed in both CLI output and Streamlit charts to demonstrate how context-aware coding approaches the conditional entropy bound.
