"""
FinQA answer parser for the RQ3 best-of-N evaluation.

With FinQA question, take a free-form LLM answer,
pull out the number it is claiming, and
decide whether it matches FinQA's gold `exe_ans`.

`grade()` returns how a match was reached, not just whether.

The three leniencies:
  1. percent scaling   -- accept pred == gold*100 or gold/100
  2. sign insensitivity -- OFF by default; FinQA sign conventions are real
  3. magnitude scaling  -- OFF by default; accept pred == gold*1e3/1e6/1e9 (units mismatch)

GENERATOR PROMPT
----------------
Instruct the generator to end with a line "Answer: <value>".
The parser falls back to last-number-in-text when the marker is missing
"""

import re
from collections import Counter
from dataclasses import dataclass, asdict

# ------------------------------------------------------------------ tunables

REL_TOL = 0.005      # 0.5% relative tolerance
ABS_TOL = 1e-6       # floor, for golds at or near zero

ALLOW_PERCENT_SCALE = True
ALLOW_SIGN_FLIP = False
ALLOW_MAGNITUDE_SCALE = False

# Magnitude factors for the units-mismatch rule is off by default.
# Applied numerically, the parser does not read scale WORDS.
# A "in millions" in a table header says nothing reliable about the
# units of the figure the model actually reports.

MAGNITUDE_FACTORS = (1e3, 1e6, 1e9)

# answer formatting, in order of priority
ANSWER_MARKERS = [
    re.compile(r"(?:final\s+)?answer\s*(?:is)?\s*[:\-]\s*(.+?)(?:\n|$)"),
    re.compile(r"\*\*answer\*\*\s*[:\-]?\s*(.+?)(?:\n|$)"),
    re.compile(r"the\s+answer\s+is\s+(.+?)(?:\n|\.|$)"),
]

# $ 1,234.56 | (123) | 45.6% | -12 | 1.2
NUMBER_RE = re.compile(
    r"""
    (?P<neg_paren>\()? # accounting negative ()
    \s*\$?\s* # space and optional dollar sign
    (?P<sign>[-+])? # optional sign - +
    (?P<num>\d[\d,]*(?:\.\d+)?) # number with optional decimal
    \s* # optional space
    (?P<close>\))? # closing paren for accounting negative
    \s* # optional space
    (?P<pct>%)? # optional percent sign
    """,
    re.VERBOSE, # ignore whitespace and comments in the regex
)

# extract yes/no from text, case-insensitive
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


# ------------------------------------------------------------------ dataclass

# result of grading one response against one gold answer
@dataclass
class Grade:
    correct: bool
    # how the match was reached
    match_type: str # exact | percent_scale | magnitude_scale | sign_flip | yesno | no_number | mismatch
    # where prediction came from
    extraction: str # marker | last_number | yesno | none
    pred_value: float | None
    gold_value: float | None

    def row(self):
        return asdict(self) # convert dataclass to dict for JSON serialization


# ------------------------------------------------------------number extraction

# match one NUMBER_RE match to a float, honouring $, commas, (), %, sign.
def _to_float(m: re.Match) -> float | None:
    """Convert one NUMBER_RE match to a float, honouring $, commas, (), %, sign."""
    raw = m.group("num").replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    if m.group("neg_paren") and m.group("close"):
        val = -val
    if m.group("sign") == "-":
        val = -val
    return val


def _numbers_in(text: str) -> list[tuple[float, bool]]:
    """All numbers in `text` as (value, boolean percent)."""
    out = []
    for m in NUMBER_RE.finditer(text):
        val = _to_float(m)
        if val is not None:
            out.append((val, bool(m.group("pct"))))
    return out


# extract the first number after an answer marker, or the last number in the text
def extract(text: str) -> tuple[float | None, bool, str]:
    """
    Return (value, is_percent, how) where how is one of:
    'marker' | 'last_number' | 'yesno' | 'none'.

    Priority: an explicit "Answer:" line beats trailing prose, because the
    model often restates intermediate figures after its conclusion.
    """
    if not text or not text.strip():
        return None, False, "none"

    low = text.lower()

    # answer formatting markers, in order of priority
    for pat in ANSWER_MARKERS:
        m = pat.search(low)
        if m:
            tail = m.group(1) # the text after the marker, which may contain a number
            nums = _numbers_in(tail) # convert all numbers in the tail to floats
            if nums:
                val, pct = nums[0]
                return val, pct, "marker"
            if YES_NO_RE.search(tail):
                return None, False, "yesno"

    nums = _numbers_in(text)
    if nums:
        val, pct = nums[-1]        # last number in the whole response
        return val, pct, "last_number"

    if YES_NO_RE.search(text):
        return None, False, "yesno"

    return None, False, "none"

def _yesno_in(text: str) -> str | None:
    """Last standalone yes/no in `text`, lowercased, or None."""
    found = YES_NO_RE.findall(text)
    return found[-1].lower() if found else None


def extract_yesno(text: str) -> tuple[str | None, str]:
    """
    yes/no counterpart of extract(): an explicit Answer-marker line beats
    trailing prose. Returns (said, how) with how in 'marker'|'last'|'none'.
    """
    if not text or not text.strip():
        return None, "none"
    low = text.lower()
    for pat in ANSWER_MARKERS:
        m = re.search(pat, low)
        if m:
            said = _yesno_in(m.group(1))
            if said:
                return said, "marker"
    said = _yesno_in(text)
    return (said, "last") if said else (None, "none")


# ---------------------------------------------------main grading function

def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * abs(b))


# the main grading function based on provided response and gold answer
def grade(
    response: str, # free-form LLM answer, usually a string
    gold, # known correct FinQA answer, usually a float, occasionally 'yes'/'no'
    allow_percent_scale: bool = ALLOW_PERCENT_SCALE,
    allow_sign_flip: bool = ALLOW_SIGN_FLIP,
    allow_magnitude_scale: bool = ALLOW_MAGNITUDE_SCALE,
) -> Grade:
    """
    `gold` is FinQA's exe_ans: usually a float, occasionally 'yes'/'no'.
    """

    # above extract function to parse the model's response to get a predicted value
    pred, pred_is_pct, how = extract(response)

    # --- yes/no items -----------------------------------------------------
    if isinstance(gold, str) and gold.strip().lower() in {"yes", "no"}:
        said, _ = extract_yesno(response or "")
        return Grade(
            correct=(said == gold.strip().lower()),
            match_type="yesno" if said else "no_number",
            extraction="yesno" if said else "none",
            pred_value=None, gold_value=None,
        )

    try:
        gold_f = float(gold)
    # if gold is not a number, nor yes/no
    except (TypeError, ValueError):
        return Grade(False, "no_number", how, pred, None)

    # if no prediction
    if pred is None:
        return Grade(False, "no_number", how, None, gold_f)

    # from above function, if within tolerance, exact match
    if _close(pred, gold_f):
        return Grade(True, "exact", how, pred, gold_f)

    # --- percentage scale -------------------------------------------------
    # FinQA stores percentage answers as 0.141 and as 14.1.
    # This is the most common source of incorrect failures.
    # Adjusting, to ensure accuracy is not under-reported.

    if allow_percent_scale:
        if _close(pred, gold_f * 100) or _close(pred, gold_f / 100):
            return Grade(True, "percent_scale", how, pred, gold_f)

    # --- magnitude ($ vs $m vs $bn) --------------------------------------
    # set to False by default, because FinQA's golds should be in the right units.
    # used in self-tests to show that the parser can be made more lenient if needed.
    if allow_magnitude_scale:
        for k in MAGNITUDE_FACTORS:
            if _close(pred, gold_f * k) or _close(pred, gold_f / k):
                return Grade(True, "magnitude_scale", how, pred, gold_f)

    # --- sign flip --------------------------------------------------------
    # set to False by default, because FinQA's golds should be in the right sign.
    # used in self-tests to show that the parser can be made more lenient if needed.
    if allow_sign_flip and _close(abs(pred), abs(gold_f)):
        return Grade(True, "sign_flip", how, pred, gold_f)

    return Grade(False, "mismatch", how, pred, gold_f)


# ------------------------------------------------------------------ reporting

def summarise(grades: list[Grade]) -> dict:
    n = len(grades)
    if n == 0:
        return {}
    correct = sum(g.correct for g in grades)
    return {
        "n": n,
        "accuracy": correct / n,
        "match_types": dict(Counter(g.match_type for g in grades)),
        "extraction": dict(Counter(g.extraction for g in grades)),
        "pct_via_percent_rule": sum(
            g.match_type == "percent_scale" for g in grades) / max(correct, 1),
        "pct_no_number_found": sum(
            g.extraction == "none" for g in grades) / n,
    }

# ---------------------------------------------------------------- self-tests

if __name__ == "__main__":
    cases = [
        # (response, gold, expect_correct, expect_match_type)
        ("The change was 14.1%.\nAnswer: 14.1%", 14.1, True, "exact"),
        ("Answer: 0.141", 14.1, True, "percent_scale"),
        ("...so we get 14.1 percent.\nAnswer: 14.1", 0.141, True, "percent_scale"),
        ("Answer: $ 1,234.56", 1234.56, True, "exact"),
        ("Answer: (123)", -123.0, True, "exact"),
        ("Answer: -123", -123.0, True, "exact"),
        ("Answer: 123", -123.0, False, "mismatch"),
        ("Revenue rose. Answer: 2.5", 2.5001, True, "exact"),      # within 0.5%
        ("Answer: 2.6", 2.5, False, "mismatch"),                   # outside
        
        # marker must beat trailing prose
        ("Answer: 42\nNote that 2019 revenue was 8,900.", 42.0, True, "exact"),

        # no marker -> falls back to last number
        ("First 100, then 200, giving 300.", 300.0, True, "exact"),
        ("I cannot determine this.", 5.0, False, "no_number"),
        ("Answer: yes", "yes", True, "yesno"),
        ("Answer: no", "yes", False, "yesno"),
        ("Answer: 0", 0.0, True, "exact"),

        # yes/no: marker beats a stray yes/no in the prose (regression for extract_yesno)
        ("No change was needed. Answer: yes", "yes", True, "yesno")
    ]

    # collect grades for all cases, report fails, and check extraction-priority
    grades, fails = [], 0
    for resp, gold, want_ok, want_type in cases:
        g = grade(resp, gold)
        grades.append(g)
        ok = (g.correct == want_ok) and (g.match_type == want_type)
        if not ok:
            fails += 1
            print(f"FAIL  {resp[:44]!r:48s} gold={gold}")
            print(f"      got correct={g.correct} type={g.match_type} "
                  f"pred={g.pred_value} via={g.extraction}")

    print(f"\n{len(cases) - fails}/{len(cases)} self-tests passed")

    # extraction-priority check
    v, _, how = extract("Answer: 42\nNote that 2019 revenue was 8,900.")
    assert (v, how) == (42.0, "marker"), (v, how)

    # magnitude rule is off by default, on when asked
    assert grade("Answer: 1200", 1.2).correct is False
    assert grade("Answer: 1200", 1.2, allow_magnitude_scale=True).correct is True

    print("summary on self-test set:")
    for k, val in summarise(grades).items():
        print(f"  {k}: {val}")
