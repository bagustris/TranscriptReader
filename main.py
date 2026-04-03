#!/usr/bin/env python3
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import fitz  # pymupdf
from PIL import Image
from openai import OpenAI

from pipeline import run_pipeline, run_single_shot, run_vlm_single_shot, run_vlm_hf_single_shot, run_ollama_vlm, parse_transcript

OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "deepseek-r1:latest"
DEFAULT_VLM_MODEL = "datalab-to/chandra"


def _vllm_is_ready(base_url: str) -> bool:
    """Return True if a vLLM server is already responding at base_url."""
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2)
        return True
    except Exception:
        return False


def _build_vllm_cmd(model: str) -> list[str]:
    """Build a vLLM server launch command.

    Prefers the native ``vllm`` binary when available; falls back to the
    Docker-based launcher bundled with chandra-ocr.
    """
    import shutil

    if shutil.which("vllm"):
        return ["vllm", "serve", model]

    # Fall back to Docker (mirrors chandra.scripts.vllm)
    from chandra.settings import settings as _cs

    return [
        "sudo",
        "docker",
        "run",
        "--runtime",
        "nvidia",
        "--gpus",
        f"device={_cs.VLLM_GPUS}",
        "-v",
        f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "--env",
        "VLLM_ATTENTION_BACKEND=TORCH_SDPA",
        "-p",
        "8000:8000",
        "--ipc=host",
        "vllm/vllm-openai:latest",
        "--model",
        model,
        "--no-enforce-eager",
        "--max-num-seqs",
        "32",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max_num_batched_tokens",
        "65536",
        "--gpu-memory-utilization",
        ".9",
        "--served-model-name",
        _cs.VLLM_MODEL_NAME,
    ]


def ensure_vllm(model: str, base_url: str) -> subprocess.Popen | None:
    """Start vLLM server if not already running. Returns the process or None."""
    if _vllm_is_ready(base_url):
        return None

    cmd = _build_vllm_cmd(model)
    print(f"Starting vLLM server for {model}...", file=sys.stderr, flush=True)
    print(f"  command: {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 5 minutes for the server to become ready
    for elapsed in range(300):
        if proc.poll() is not None:
            print("error: vLLM process exited unexpectedly.", file=sys.stderr)
            sys.exit(1)
        if _vllm_is_ready(base_url):
            print(f"vLLM server ready ({elapsed + 1}s).", file=sys.stderr, flush=True)
            return proc
        if elapsed % 15 == 0:
            print(f"  waiting for vLLM... ({elapsed}s)", file=sys.stderr, flush=True)
        time.sleep(1)

    proc.terminate()
    print("error: vLLM server did not become ready within 5 minutes.", file=sys.stderr)
    sys.exit(1)


def _ollama_unload_all() -> None:
    """Unload all cached Ollama models from GPU memory (set keep_alive=0)."""
    try:
        import json as _json
        req_data = _json.dumps({"model": "", "keep_alive": 0}).encode()
        # Ask Ollama for the list of running models
        with urllib.request.urlopen(
            OLLAMA_BASE_URL.replace("/v1", "") + "/api/ps", timeout=3
        ) as resp:
            ps = _json.loads(resp.read())
        for m in ps.get("models", []):
            name = m.get("name", "")
            if not name:
                continue
            req = urllib.request.Request(
                OLLAMA_BASE_URL.replace("/v1", "") + "/api/generate",
                data=_json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=10)
                print(f"Unloaded Ollama model: {name}", file=sys.stderr, flush=True)
            except Exception:
                pass
    except Exception:
        pass  # Ollama not running or no models loaded — harmless


def pdf_to_images(path: str) -> list:
    """Convert every PDF page to a base64-encoded PNG string."""
    doc = fitz.open(path)
    images = []
    for page in doc:
        mat = fitz.Matrix(2, 2)  # 2x zoom for legible resolution
        pix = page.get_pixmap(matrix=mat)
        images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return images


def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF. Uses cached OCR if available, otherwise runs Chandra-OCR.

    On first run the OCR output is saved to <path>.ocr.md so subsequent calls
    return the same stable text without re-running the model.  Delete the cache
    file to force a fresh OCR pass.
    """
    cache_path = path + ".ocr.md"
    if os.path.exists(cache_path):
        print(f"Using cached OCR: {cache_path}", file=sys.stderr, flush=True)
        with open(cache_path) as f:
            return f.read()

    doc = fitz.open(path)
    text_pages = []
    image_pages = []

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_pages.append((i, text))
        else:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_pages.append((i, img))
    doc.close()

    if not image_pages:
        return "\n".join(t for _, t in text_pages)

    # Use Chandra-OCR for image pages
    print("Loading Chandra-OCR model...", file=sys.stderr, flush=True)
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem

    manager = InferenceManager(method="hf")
    print("Running Chandra-OCR...", file=sys.stderr, flush=True)
    results = []
    for idx, (_, img) in enumerate(image_pages):
        print(
            f"  OCR page {idx + 1}/{len(image_pages)}...", file=sys.stderr, flush=True
        )
        page_results = manager.generate(
            [BatchInputItem(image=img, prompt_type="ocr_layout")]
        )
        results.extend(page_results)

    # Free GPU memory so downstream LLM calls (Ollama) can use it
    import gc
    import torch
    del manager
    gc.collect()
    torch.cuda.empty_cache()
    print("Chandra-OCR done, GPU memory freed.", file=sys.stderr, flush=True)

    # Merge text and OCR results in page order
    all_pages = {}
    for i, text in text_pages:
        all_pages[i] = text
    for (i, _), result in zip(image_pages, results):
        all_pages[i] = result.markdown

    ocr_text = "\n".join(all_pages[i] for i in sorted(all_pages))
    with open(cache_path, "w") as f:
        f.write(ocr_text)
    print(f"Saved OCR cache: {cache_path}", file=sys.stderr, flush=True)
    return ocr_text


def process_file(path: str, args) -> dict:
    """Run the selected mode on a single PDF and return the result dict."""
    if args.mode == "ocr-parse":
        _ollama_unload_all()
        print(f"Running chandra-OCR + Python table parser on {path}...", file=sys.stderr, flush=True)
        ocr_text = extract_text_from_pdf(path)
        return parse_transcript(ocr_text)
    elif args.mode == "ollama-vlm":
        page_images = pdf_to_images(path)
        model = args.model or "gemma3:12b"
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=600)
        return run_ollama_vlm(client, model, page_images)
    elif args.mode == "vlm-hf":
        doc = fitz.open(path)
        pil_images = []
        for page in doc:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            pil_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        return run_vlm_hf_single_shot(pil_images)
    elif args.mode == "vlm":
        page_images = pdf_to_images(path)
        model = args.model or DEFAULT_VLM_MODEL
        ensure_vllm(model, args.vlm_url)
        client = OpenAI(base_url=args.vlm_url, api_key="vllm")
        return run_vlm_single_shot(client, model, page_images)
    else:
        _ollama_unload_all()
        transcript_text = extract_text_from_pdf(path)
        model = args.model or DEFAULT_MODEL
        if args.openai:
            client = OpenAI()
        else:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=600)
        if args.mode == "single-shot":
            return run_single_shot(client, model, transcript_text)
        else:
            return run_pipeline(client, model, transcript_text)


def main():
    parser = argparse.ArgumentParser(description="LLM-based transcript analyzer")
    parser.add_argument("file", help="Path to transcript PDF file or folder of PDFs")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "single-shot", "vlm", "vlm-hf", "ollama-vlm", "ocr-parse"],
        default="pipeline",
        help="Run multi-step pipeline, single-shot, VLM via vLLM, VLM via HF, VLM via Ollama, "
             "or direct OCR table parsing (default: pipeline)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Model to use (default: {DEFAULT_MODEL} for text modes, {DEFAULT_VLM_MODEL} for vlm)",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API instead of local Ollama",
    )
    parser.add_argument(
        "--vlm-url",
        default=VLLM_BASE_URL,
        help=f"vLLM server base URL for --mode vlm (default: {VLLM_BASE_URL})",
    )
    args = parser.parse_args()

    if os.path.isdir(args.file):
        import csv
        import glob as _glob
        pdf_paths = sorted(_glob.glob(os.path.join(args.file, "*.pdf")))
        if not pdf_paths:
            print(f"error: no PDF files found in {args.file}", file=sys.stderr)
            sys.exit(1)
        writer = csv.DictWriter(sys.stdout, fieldnames=["file", "S", "A", "B", "C", "Pass"])
        writer.writeheader()
        for path in pdf_paths:
            result = process_file(path, args)
            writer.writerow({
                "file": os.path.basename(path),
                "S": result.get("S", 0),
                "A": result.get("A", 0),
                "B": result.get("B", 0),
                "C": result.get("C", 0),
                "Pass": result.get("Pass", 0),
            })
    else:
        result = process_file(args.file, args)
        json.dump(result, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
