# TranscriptReader

Analyzes academic transcript PDFs and outputs the total credits earned per grade category (S / A / B / C / Pass).

```json
{"S": 6.0, "A": 12.0, "B": 9.0, "C": 3.0, "Pass": 2.0}
```

---

## How it works

Three modes are available:

| Mode | Input to LLM | Best for |
|------|-------------|----------|
| `pipeline` (default) | Extracted text, 4 chained steps | Accuracy, debuggability |
| `single-shot` | Extracted text, 1 prompt | Speed |
| `vlm` | Raw PDF page images | Scanned/image PDFs with no selectable text |

### `pipeline` and `single-shot` (text modes)

```
PDF → text extraction → LLM (Ollama or OpenAI) → JSON
```

Text is extracted with PyMuPDF. If a page has no selectable text (scanned), it falls back to the Chandra OCR model locally via HuggingFace.

### `vlm` (vision mode)

```
PDF → page images (PNG) → Chandra VLM via vLLM → JSON
```

Each page is rendered at 2× resolution and sent directly to a vision-language model. The model reads and analyzes the transcript in one pass — no separate OCR step needed.

---

## Grade mapping rules

| Source format | S | A | B | C | Pass |
|--------------|---|---|---|---|------|
| 100-point | 90–100 | 80–89 | 70–79 | 60–69 | — |
| Letter (standard) | A+ | A / A− | B+ / B | B− / C+ / C / D | Pass |
| 3-stage (A/B/C) | A | A | B | C | — |
| 4-stage (S/A/B/C) | S | A | B | C | — |
| Indonesian 5-stage | A | AB | B | BC / C | — |
| 20-point (French) | 16.0–20.0 | 14.0–15.99 | 12.0–13.99 | 10.0–11.99 | — |

Failed subjects (below the passing threshold) are excluded from all counts.

---

## Installation

### Requirements

- Python 3.10+
- (For `vlm` mode) A GPU with enough VRAM to serve Chandra 9B — roughly 20 GB on a single A100/H100, or across multiple smaller GPUs.

### Install dependencies

```bash
pip install openai pymupdf Pillow
```

For text modes with scanned pages (Chandra local fallback):

```bash
pip install chandra-ocr
```

For `vlm` mode — install and start a vLLM server:

```bash
pip install vllm
vllm serve datalab-to/chandra
```

The model weights are downloaded automatically by vLLM on first run, or you can pre-download them:

```bash
pip install huggingface_hub
huggingface-cli download datalab-to/chandra
# weights cached at ~/.cache/huggingface/hub/models--datalab-to--chandra/
```

---

## Usage

```bash
python main.py <transcript.pdf> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `pipeline` | `pipeline`, `single-shot`, or `vlm` |
| `-m / --model` | see below | Model name to use |
| `--openai` | off | Use OpenAI API instead of local Ollama |
| `--vlm-url` | `http://localhost:8000/v1` | vLLM server URL (for `--mode vlm`) |

**Default models:**
- Text modes (`pipeline`, `single-shot`): `deepseek-r1:latest` via Ollama
- VLM mode: `datalab-to/chandra` via vLLM

### Examples

```bash
# Default: multi-step pipeline with local Ollama
python main.py transcript.pdf

# Single-shot with a different Ollama model
python main.py transcript.pdf --mode single-shot -m llama3.1:8b

# Use OpenAI API
python main.py transcript.pdf --openai -m gpt-4o

# VLM mode with Chandra (default) — requires vLLM server running
python main.py transcript.pdf --mode vlm

# VLM mode with a custom model or remote server
python main.py transcript.pdf --mode vlm -m allenai/olmOCR-2-0924-hf
python main.py transcript.pdf --mode vlm --vlm-url http://gpu-server:8000/v1
```

---

## Project structure

```
transcript_reader/
├── main.py          # CLI entry point, PDF → images/text, dispatches to pipeline
├── pipeline.py      # LLM call logic: pipeline steps + VLM single-shot
├── prompts.py       # All prompt templates (extraction, validation, grading, aggregation)
├── requirements.txt # Python dependencies
└── designs/         # Design notes for each pipeline step
    ├── 01_llm_extraction.md
    ├── 02_llm_prompt_template.md
    ├── 03_output_validation_prompt.md
    ├── 04_grading_mapping.md
    ├── 05_credit_aggregation.md
    └── 06_full_single_shot.md
```

### Pipeline steps (text modes)

1. **Extract** — pull course name, credit, and score from raw transcript text
2. **Validate** — check and auto-correct the extracted JSON
3. **Map grades** — convert scores/letters to S/A/B/C/Pass categories
4. **Aggregate** — sum credits per category

VLM mode collapses all four steps into a single model call.

---

## About Chandra

[Chandra](https://huggingface.co/datalab-to/chandra) is a 9B vision-language model by Datalab, built on Qwen3-VL. It leads the OlmOCR benchmark (83.1 ±0.9) and supports 40+ languages. It handles tables, handwriting, and complex layouts natively.

License: Apache 2.0 (code) / modified OpenRAIL-M (weights) — free for research and personal use.
