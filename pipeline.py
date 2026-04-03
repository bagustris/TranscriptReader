import html as _html_module
import html.parser as _html_parser
import json
import re
import sys
from openai import OpenAI

import prompts


def parse_transcript(ocr_markdown: str) -> dict:
    """Parse ITS-format HTML table from chandra OCR markdown; return credit totals.

    Handles the ITS bilingual transcript layout where:
    - S column = semester number (ignore it)
    - K column = SKS/credit value (use this)
    - Grades: A→A, AB→A, B→B, BC→C, C→C, D/E→exclude
    - Duplicate course codes → keep first occurrence only
    - Merged-cell course groups (Final Project, Praktikum Sistem) → count once
    """

    class _Parser(_html_parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self._in_tbody = False
            self._in_tr = False
            self._in_cell = False
            self._cur_row: list[str] = []
            self._cur_cell: list[str] = []
            self.rows: list[list[str]] = []

        def handle_starttag(self, tag, attrs):
            if tag == "tbody":
                self._in_tbody = True
            elif tag == "tr" and self._in_tbody:
                self._in_tr = True
                self._cur_row = []
            elif tag in ("td", "th") and self._in_tr:
                self._in_cell = True
                self._cur_cell = []

        def handle_endtag(self, tag):
            if tag == "tbody":
                self._in_tbody = False
            elif tag == "tr" and self._in_tbody:
                self._in_tr = False
                self.rows.append(self._cur_row[:])
            elif tag in ("td", "th") and self._in_tr:
                self._in_cell = False
                self._cur_row.append("".join(self._cur_cell).strip())

        def handle_data(self, data):
            if self._in_cell:
                self._cur_cell.append(data)

        def handle_entityref(self, name):
            if self._in_cell:
                self._cur_cell.append(_html_module.unescape(f"&{name};"))

        def handle_charref(self, name):
            if self._in_cell:
                self._cur_cell.append(_html_module.unescape(f"&#{name};"))

    parser = _Parser()
    parser.feed(ocr_markdown)

    # ITS 5-stage grade → output category
    GRADE_MAP = {"A": "A", "AB": "A", "B": "B", "BC": "C", "C": "C", "Pass": "Pass"}
    LETTER_MAP = {"A+": "S", "A-": "A", "B+": "B", "B-": "C", "C+": "C"}
    EXCLUDED = {"D", "E"}

    totals: dict[str, float] = {"S": 0.0, "A": 0.0, "B": 0.0, "C": 0.0, "Pass": 0.0}
    seen_codes: set[str] = set()
    group_seen: set[str] = set()

    for row in parser.rows:
        # Each table row has up to 3 groups of 7 cells:
        # No | Kode | Mata Kuliah | S (semester) | K (credits) | Nilai (grade) | Grade
        for g in range(3):
            offset = g * 7
            if offset + 7 > len(row):
                break
            code  = row[offset + 1]
            name  = row[offset + 2]
            k_str = row[offset + 4]
            grade = row[offset + 5]

            if not code or not k_str or not grade:
                continue
            if code in ("Kode", "No"):
                continue

            # Skip exact code duplicates (bilingual same-code entries)
            if code in seen_codes:
                continue
            seen_codes.add(code)

            if grade in EXCLUDED:
                continue

            # Merged-cell groups: treat all rows sharing a group name as one course
            group_key = None
            if "Final Project" in name:
                group_key = "Final Project"
            elif "Praktikum Sistem" in name:
                group_key = "Praktikum Sistem"
            if group_key:
                if group_key in group_seen:
                    continue
                group_seen.add(group_key)

            try:
                credit = float(k_str)
            except ValueError:
                continue

            if grade in GRADE_MAP:
                category = GRADE_MAP[grade]
            else:
                category = LETTER_MAP.get(grade, "C")

            totals[category] += credit

    return totals


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from deepseek-r1 responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Extract JSON from a response that may contain markdown fences or extra text."""
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _clean(raw: str) -> str:
    return _extract_json(_strip_think_tags(raw))


def _chat(client: OpenAI, model: str, system: str, user: str) -> str:
    print(f"  [{model}] calling...", file=sys.stderr, flush=True)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return _clean(resp.choices[0].message.content.strip())


def extract_courses(client: OpenAI, model: str, transcript_text: str) -> str:
    """Step 1 (designs 01+02): Extract structured course data from raw transcript."""
    user_msg = prompts.EXTRACTION_USER.format(transcript_text=transcript_text)
    return _chat(client, model, prompts.EXTRACTION_SYSTEM, user_msg)


def validate_output(client: OpenAI, model: str, llm_output: str) -> str:
    """Step 2 (design 03): Validate / correct the extracted JSON."""
    user_msg = prompts.VALIDATION_USER.format(llm_output=llm_output)
    result = _chat(client, model, prompts.VALIDATION_SYSTEM, user_msg)
    if result == "VALID":
        return llm_output
    return result


def map_grades(client: OpenAI, model: str, structured_json: str) -> str:
    """Step 3 (design 04): Convert scores to S/A/B/C/Pass categories."""
    user_msg = prompts.GRADING_USER.format(structured_json=structured_json)
    return _chat(client, model, prompts.GRADING_SYSTEM, user_msg)


def aggregate_credits(client: OpenAI, model: str, categorized_json: str) -> dict:
    """Step 4 (design 05): Sum credits per category."""
    user_msg = prompts.AGGREGATION_USER.format(categorized_json=categorized_json)
    raw = _chat(client, model, prompts.AGGREGATION_SYSTEM, user_msg)
    return json.loads(raw)


def run_pipeline(client: OpenAI, model: str, transcript_text: str) -> dict:
    """Run the full multi-step pipeline: extract → validate → grade → aggregate."""
    print("Step 1/4: Extracting courses...", file=sys.stderr, flush=True)
    extracted = extract_courses(client, model, transcript_text)
    print("Step 2/4: Validating output...", file=sys.stderr, flush=True)
    validated = validate_output(client, model, extracted)
    print("Step 3/4: Mapping grades...", file=sys.stderr, flush=True)
    graded = map_grades(client, model, validated)
    print("Step 4/4: Aggregating credits...", file=sys.stderr, flush=True)
    return aggregate_credits(client, model, graded)


def run_single_shot(client: OpenAI, model: str, transcript_text: str) -> dict:
    """Run the single-shot prompt (design 06) that does everything at once."""
    print("Running single-shot analysis...", file=sys.stderr, flush=True)
    user_msg = prompts.SINGLE_SHOT_USER.format(transcript_text=transcript_text)
    raw = _chat(client, model, prompts.SINGLE_SHOT_SYSTEM, user_msg)
    return json.loads(raw)


def run_ollama_vlm(client: OpenAI, model: str, page_images_b64: list) -> dict:
    """Use a vision-capable Ollama model to analyze transcript images directly."""
    print(f"  [{model}] analyzing {len(page_images_b64)} page(s)...", file=sys.stderr, flush=True)
    content = []
    for img_b64 in page_images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    content.append({"type": "text", "text": prompts.VLM_SINGLE_SHOT_USER})
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts.SINGLE_SHOT_SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )
    raw = _clean(resp.choices[0].message.content.strip())
    return json.loads(raw)


def run_vlm_hf_single_shot(images: list) -> dict:
    """Use Chandra (HF mode) as a VLM to analyze transcript images directly."""
    import gc, json, torch
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem

    print(f"Loading Chandra VLM (HF mode)...", file=sys.stderr, flush=True)
    manager = InferenceManager(method="hf")
    print(f"  Analyzing {len(images)} page(s)...", file=sys.stderr, flush=True)

    results = manager.generate([
        BatchInputItem(image=img, prompt=prompts.VLM_SINGLE_SHOT_USER)
        for img in images
    ])

    del manager
    gc.collect()
    torch.cuda.empty_cache()

    # Sum each category across all pages
    totals: dict = {"S": 0.0, "A": 0.0, "B": 0.0, "C": 0.0, "Pass": 0.0}
    for r in results:
        page_data = json.loads(_clean(r.raw))
        for k in totals:
            totals[k] += float(page_data.get(k, 0.0))
    return totals


def run_vlm_single_shot(client: OpenAI, model: str, page_images: list) -> dict:
    """Send all PDF page images directly to a VLM and get grade counts in one shot."""
    print(f"Running VLM single-shot ({len(page_images)} page(s))...", file=sys.stderr, flush=True)

    content = []
    for img_b64 in page_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    content.append({"type": "text", "text": prompts.VLM_SINGLE_SHOT_USER})

    print(f"  [{model}] calling VLM...", file=sys.stderr, flush=True)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts.SINGLE_SHOT_SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_tokens=1000,
    )
    raw = _clean(resp.choices[0].message.content.strip())
    return json.loads(raw)
