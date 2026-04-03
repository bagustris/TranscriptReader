# Prompt templates derived from designs/01–06

# --- 02: Extraction prompt template ---
EXTRACTION_SYSTEM = (
    "You are a precise data extraction engine.\n"
    "Return only valid JSON.\n"
    "No explanation."
)

EXTRACTION_USER = """\
Extract structured course data from the transcript text below.

Return JSON list:
[
  {{
    "course": "...",
    "credit": float,
    "score": string or float
  }}
]

Rules for identifying the credit column:
- If the table has a column explicitly labeled "SKS", "Credits", or "Kredit" — use that column.
- If the table has columns "S" and "K" (common in Indonesian ITS transcripts), "S" is the semester
  number (ignore it) and "K" is the SKS/credit value — use the K column as credit.
- For French/European transcripts with a 20-point scale, use the credit column as labeled.
- For 100-point or letter-grade transcripts, use the credits column as labeled.

Rules for the score field:
- Preserve letter grades exactly as they appear (e.g. "A", "AB", "B", "BC", "C", "Pass").
- Do NOT convert letter grades to numeric values.
- For numeric scores, use the number as-is.

Deduplication:
- If the same course code appears more than once, include it only ONCE (first occurrence).
- Failed courses (D, E, or numeric score < 60) should be excluded.

Transcript Text:
\"\"\"
{transcript_text}
\"\"\""""

# --- 03: Validation prompt template ---
VALIDATION_SYSTEM = "You validate JSON."

VALIDATION_USER = """\
Check whether the following JSON is valid and contains only:
[
  {{
    "course": string,
    "credit": number,
    "score": number or "Pass" or letter grade
  }}
]

If valid:
Return EXACTLY: VALID

If invalid:
Return corrected JSON only.

JSON:
{llm_output}"""

# --- 04: Grading mapping prompt template ---
GRADING_SYSTEM = "You are a grading conversion engine."

GRADING_USER = """\
Convert these course records to S/A/B/C categories using rules:

Numeric rule:
90–100 → S
80–89 → A
70–79 → B
60–69 → C

Letter rule:
A+ → S
A/A- → A
B+/B → B
B-/C+/C/D → C
"Pass" → Pass

Indonesian 5-stage rule (A, AB, B, BC, C, D, E):
A → A
AB → A
B → B
BC → C
C → C
D → C (failing, exclude if possible)
E → C (failing, exclude if possible)

20-point rule:
16.0 - 20.0 → S
14.0 - 15.99 → A
12.0 - 13.99 → B
10.0 - 11.99 → C

Input:
{structured_json}

Return:
[
  {{
    "course": "...",
    "credit": float,
    "category": "S" | "A" | "B" | "C" | "Pass"
  }}
]"""

# --- 05: Credit aggregation prompt template ---
AGGREGATION_SYSTEM = "You are a numerical aggregator."

AGGREGATION_USER = """\
Sum total credits per category.

Input:
{categorized_json}

Return JSON:
{{
  "S": float,
  "A": float,
  "B": float,
  "C": float,
  "Pass": float
}}"""

# --- 06: Full single-shot prompt template ---
SINGLE_SHOT_SYSTEM = (
    "You are a transcript analysis engine.\n"
    "Return STRICT JSON only."
)

SINGLE_SHOT_USER = """\
Analyze this transcript.

Tasks:
1. Extract courses.
2. Detect credit.
3. Detect score.
4. Convert to categories:
   90–100 → S
   80–89 → A
   70–79 → B
   60–69 → C
   A+ → S
   A/A- → A
   B+/B → B
   B-/C+/C/D → C
   Indonesian: A→A, AB→A, B→B, BC→C, C→C
   "Pass" → Pass
5. Sum total credits per category.

Return EXACT JSON:
{{
  "S": float,
  "A": float,
  "B": float,
  "C": float,
  "Pass": float
}}

Transcript:
\"\"\"
{transcript_text}
\"\"\"
"""

# --- 07: VLM single-shot (image input, no transcript_text placeholder) ---
VLM_SINGLE_SHOT_USER = """\
Analyze the academic transcript shown in the image(s).

STEP 1 — Identify the credit column:
  - Look for a column labeled "K", "SKS", "Credits", or "Kredit".
  - In Indonesian ITS-format transcripts the table has columns:
      No | Code | Course Name | S (semester — IGNORE) | K (credits/SKS — USE THIS) | Grade
  - Use the K column value as the credit for each course.
  - Count EACH PHYSICAL TABLE ROW as ONE course.
    If a row spans multiple lines (e.g. bilingual name), still count it once.
  - If the same course code appears in more than one row, count it ONCE.

STEP 2 — Map grades to categories:
  Indonesian 5-stage: A → A, AB → A, B → B, BC → C, C → C, D/E → exclude
  4-stage (S,A,B,C): map directly
  3-stage (A,B,C): A→A, B→B, C→C
  Numeric: 90–100→S, 80–89→A, 70–79→B, 60–69→C, <60→exclude
  Letter: A+→S, A/A-→A, B+/B→B, B-/C+/C/D→C, Pass→Pass
  French 20-pt: 16–20→S, 14–15.99→A, 12–13.99→B, 10–11.99→C

STEP 3 — Sum credits per category (exclude failed/D/E courses).

Return EXACT JSON only, no explanation:
{
  "S": <float>,
  "A": <float>,
  "B": <float>,
  "C": <float>,
  "Pass": <float>
}"""
