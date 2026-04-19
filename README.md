# Context-Aware Huffman Compressor/Decompressor

🌍 **Live Demo:** [https://context-aware-huffman.streamlit.app](https://context-aware-huffman.streamlit.app)

A Context-Aware Huffman Coding compressor and decompressor that improves upon standard Huffman coding by building per-context Huffman trees. Instead of relying on a single global frequency table, it tracks which tokens follow each context (the previous *N* tokens) and builds separate Huffman trees per context. This achieves better compression rates when token distributions vary significantly by context.

## Features

- **Context-Aware Coding**: Uses an *N*-order Markov model approach to build conditional Huffman trees.
- **Two Tokenization Modes**: 
  - `word` mode: Splits on whitespace boundaries, preserving whitespace tokens.
  - `char` mode: Splits text into individual characters.
- **Dynamic Context Window (`order`)**: Configurable context size (e.g., `order=1` for bigram-like context, `order=2` for trigram).
- **Streamlit Web UI**: Interactive dashboard for compressing/decompressing text and visualizing tree distributions, entropy metrics, and efficiency.
- **Command-Line Interface (CLI)**: Full-featured CLI for compressing and decompressing files directly from the terminal.

## Installation

1. Clone this repository or download the source code.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Web UI (Streamlit)

Launch the interactive web application to try out compression, decompression, and view detailed analytical charts:

```bash
streamlit run app.py
```

### 2. Command-Line Interface (CLI)

You can use `main.py` to compress and decompress files directly from your terminal.

**Compress a file:**
```bash
python main.py --compress input.txt --out output.bin --mode word --order 1
```

**Decompress a file:**
```bash
python main.py --decompress output.bin --out restored.txt
```

**Run built-in sample experiments (benchmarking):**
```bash
python main.py --experiment
```

## Architecture

- **`app.py`**: The Streamlit web interface. It handles in-memory byte compression/decompression and generates visualization charts.
- **`main.py`**: The CLI tool. Contains the canonical file I/O compression/decompression logic and benchmarking code.
- **Compression Pipeline**: 
  1. Tokenize input (word or char).
  2. Build global and per-context frequency maps.
  3. Inject an `<ESC>` fallback token for unseen context transitions.
  4. Build deterministic Huffman trees.
  5. Encode bitstream using context-specific codes (falling back to global codes via `<ESC>`).
  6. Serialize with a lightweight JSON header containing the frequency maps.

## Entropy Metrics

The tool calculates and displays:
- **Shannon Entropy (Global)**: Theoretical lower bound for standard memoryless coding.
- **Conditional Entropy**: Theoretical lower bound for context-aware predictive coding.
- **Actual Bits/Token**: The real performance of this context-aware Huffman implementation compared against standard Huffman coding.
