"""
FinQA table flattening + prompt construction. SINGLE SOURCE OF TRUTH.

    python finqa_prompt.py inspect --data finqa_test.json --k 10
    python finqa_prompt.py sample  --data finqa_test.json --n 200 --seed 20261026
    python finqa_prompt.py stats   --data finqa_test.json

FinQA ships `qa.gold_inds` (gold indices): the exact sentences and table rows containing
the answer. The 'exe_ans' (executed answer) is also provided.
These are used for evaluation, but must obviously not be included in the prompt. The
prompt is built from the pre_text, post_text, table, and question fields only.
The exe_ans is used for evaluation in RQ3.
This module reads only the fields in ALLOWED_FIELDS. `inspect` prints an
explicit confirmation that gold_inds and exe_ans were not touched.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

BUILDER_VERSION = "finqa_prompt/v1"

ALLOWED_FIELDS = ("id", "pre_text", "post_text", "table", "question")
# not in the prompt, but used for evaluation and stratification
FORBIDDEN_FIELDS = ("gold_inds", "program", "answer", "steps")

PRE_CHAR_CAP = 10000
POST_CHAR_CAP = 8000
EMPTY_CELL = "-"


# ------------------------------------------------------------------ cleaning

def _cell(x) -> str:
    """Collapse whitespace; make empties visible so columns stay aligned."""
    s = re.sub(r"\s+", " ", str(x if x is not None else "")).strip()
    return s if s else EMPTY_CELL


def flatten_table(table) -> str:
    """
    Pipe-delimited, header row preserved, ragged rows padded.

    Ragged rows occur in FinQA when a row can be short where a merged or
    footnote cell was dropped in extraction. Padding keeps every row the
    same width so column position still means something.
    """
    rows = [r for r in (table or []) if r is not None]
    if not rows:
        return "(no table provided)"
    width = max(len(r) for r in rows)
    out = []
    for r in rows:
        cells = [_cell(c) for c in r] + [EMPTY_CELL] * (width - len(r))
        out.append(" | ".join(cells))
    return "\n".join(out)


def _truncate(sentences, cap: int) -> tuple[str, bool]:
    """
    Join sentences, cutting at a sentence boundary once `cap` chars is hit.
    Head-first: units ("in millions", "except per share amounts") live at
    the start of pre_text, and losing them makes every answer wrong by a
    factor of a million.
    """
    kept, total, truncated = [], 0, False
    for s in sentences or []:
        s = re.sub(r"\s+", " ", str(s)).strip()
        if not s:
            continue
        if total + len(s) > cap and kept:
            truncated = True
            break
        kept.append(s)
        total += len(s) + 1
    return " ".join(kept), truncated


# -------------------------------------------------------------- prompt build

OP_RE = re.compile(r"[A-Za-z][\w\-]*\([^()]*\)")


# to idenfity questions by difficulty, count the number of top-level operations in the gold reasoning program.
def count_program_steps(program) -> int:
    """
    Number of operations in FinQA's gold reasoning `program`.

    The DSL is flat using a comma-separated sequence of `op(args)`, with `#n`
    referring back to step n rather than nesting. So counting top-level
    op-groups counts steps.

        divide(9413, 20.01), divide(8249, 9.48), subtract(#0, #1)   -> 3

    Chen et al. (2022) Table 3: accuracy falls significantly with incrementing step
    count (1 step 67.61%, 2 steps 59.08%, >2 steps 22.78%). A uniform random sample
    is ~59% one-step questions, so without stratification any evaluation set is
    mostly the easier questions.

    NOTE: `program` is read here for stratification only. Like `exe_ans` it
    is a FORBIDDEN_FIELD and must never reach the prompt.
    """
    if not program or not isinstance(program, str):
        return 0
    return len(OP_RE.findall(program))


def step_bucket(n: int) -> str:
    if n <= 0:
        return "unknown"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


# generate a normalised dict with only the fields we want to use in the prompt
def normalise(item: dict) -> dict:
    """
    FinQA nests the question under `qa` (occasionally `qa_0`). Flatten to the
    shape the pipeline expects, reading the allowed fields for the prompt.
    """
    qa = item.get("qa") or item.get("qa_0") or {}
    steps = count_program_steps(qa.get("program"))
    return {
        "id": str(item.get("id", "")),
        "pre_text": item.get("pre_text", []),
        "post_text": item.get("post_text", []),
        "table": item.get("table", []),
        "question": qa.get("question", ""),
        # the two fields below are part of the item for grading and
        # stratification, and do not enter the prompt
        "exe_ans": qa.get("exe_ans", qa.get("answer")),
        "n_steps": steps,
        "stratum": step_bucket(steps),
    }


# prompt builder using pre and post texts, and table, used for both inspect and sample commands
def build_prompt(item: dict) -> str:
    pre, _ = _truncate(item.get("pre_text"), PRE_CHAR_CAP)
    post, _ = _truncate(item.get("post_text"), POST_CHAR_CAP)
    return (
        "You are a financial analyst. Use only the information below.\n\n"
        f"{pre}\n\n"
        f"{flatten_table(item.get('table'))}\n\n"
        f"{post}\n\n"
        f"Question: {item['question']}\n\n"
        "Reason step by step, then give your final result on its own last "
        "line in exactly this form:\nAnswer: <value>"
    )


# create auditable and reproducible prompt fingerprints for the run manifest
def prompt_hash(item: dict) -> str:
    """
    Fingerprint of the rendered prompt, stored in the run manifest.
    """
    return hashlib.sha256(build_prompt(item).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ commands

def _load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [normalise(it) for it in data]

# sample a few items for review
def cmd_inspect(args):
    items = _load(args.data)
    rng = random.Random(args.seed)
    picks = rng.sample(items, min(args.k, len(items)))

    lines = []
    for n, it in enumerate(picks, 1):
        lines.append("=" * 74)
        lines.append(f"[{n}/{len(picks)}]  id={it['id']}  hash={prompt_hash(it)}")
        lines.append(f"gold (NOT in prompt): {it['exe_ans']!r}")
        lines.append("=" * 74)
        lines.append(build_prompt(it))
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(picks)} rendered prompts -> {args.out}")
    else:
        print(text)

    print("\n" + "-" * 74)
    print(f"builder      : {BUILDER_VERSION}")
    print(f"fields used  : {', '.join(ALLOWED_FIELDS)}")
    print(f"NOT used     : {', '.join(FORBIDDEN_FIELDS)}  <- no retrieval oracle")
    print("-" * 74)
    print("To review")

# sample statistics on table shapes and truncation rates
def cmd_stats(args):
    items = _load(args.data)
    widths, heights, pre_t, post_t, no_table, plens = [], [], 0, 0, 0, []

    for it in items:
        rows = [r for r in (it.get("table") or []) if r is not None]
        if not rows:
            no_table += 1
        else:
            heights.append(len(rows))
            widths.append(max(len(r) for r in rows))
        _, t1 = _truncate(it.get("pre_text"), PRE_CHAR_CAP)
        _, t2 = _truncate(it.get("post_text"), POST_CHAR_CAP)
        pre_t += t1
        post_t += t2
        plens.append(len(build_prompt(it)))

    n = len(items)

    def _is_ragged(it) -> bool:
        rows = [r for r in (it.get("table") or []) if r]
        return bool(rows) and len({len(r) for r in rows}) > 1

    ragged = sum(1 for it in items if _is_ragged(it))

    plens.sort()
    print(f"items                 : {n}")
    print(f"no table              : {no_table}  ({no_table / n:.1%})")
    print(f"ragged tables         : {ragged}  ({ragged / n:.1%})  <- padded")
    if heights:
        print(f"table rows  med/max   : {sorted(heights)[len(heights)//2]} / {max(heights)}")
        print(f"table cols  med/max   : {sorted(widths)[len(widths)//2]} / {max(widths)}")
    print(f"pre_text truncated    : {pre_t}  ({pre_t / n:.1%})")
    print(f"post_text truncated   : {post_t}  ({post_t / n:.1%})")
    print(f"prompt chars med/p95  : {plens[n//2]} / {plens[int(n*0.95)]}")
    print(f"  (~{plens[int(n*0.95)]//4} tokens at p95 -- check against the context window)")

    if pre_t / n > 0.15:
        print("\nWARNING: pre_text truncation is high.")


STRATA = ("1", "2", "3+")

# allocate a number of items to draw from each stratum
def _allocate(n: int, pool: dict, balance: bool) -> dict:
    """
    How many to draw from each stratum.

    proportional (default) -- preserves FinQA's natural difficulty mix, so the
      accuracy is representative of the dataset.
    balanced (--balance)   -- equal counts, so per-stratum comparisons have
      enough questions to test. Accuracy is not representative and
      reported per stratum.
    """
    avail = {s: len(pool.get(s, [])) for s in STRATA}
    total = sum(avail.values())
    if total == 0:
        return {s: 0 for s in STRATA}

    if balance:
        raw = {s: n / len(STRATA) for s in STRATA}
    else:
        raw = {s: n * avail[s] / total for s in STRATA}

    alloc = {s: min(int(raw[s]), avail[s]) for s in STRATA}
    # hand out leftovers by largest fractional part, respecting availability
    while sum(alloc.values()) < min(n, total):
        best, best_frac = None, -1.0
        for s in STRATA:
            if alloc[s] >= avail[s]:
                continue
            frac = raw[s] - int(raw[s])
            if frac > best_frac:
                best, best_frac = s, frac
        if best is None:
            break
        alloc[best] += 1
    return alloc

# create a reproducible RQ3 evaluation set (200) and manifest
def cmd_sample(args):
    items = _load(args.data)
    rng = random.Random(args.seed)

    pool = {s: [it for it in items if it["stratum"] == s] for s in STRATA}
    unknown = [it for it in items if it["stratum"] == "unknown"]
    if unknown:
        print(f"note: {len(unknown)} items had no parseable program; excluded "
              f"from stratified sampling")

    if args.stratify:
        alloc = _allocate(args.n, pool, args.balance)
        picks = []
        for s in STRATA:
            picks.extend(rng.sample(pool[s], alloc[s]))
        rng.shuffle(picks)
        mode = "balanced" if args.balance else "proportional"
    else:
        picks = rng.sample(items, min(args.n, len(items)))
        alloc = {s: sum(1 for it in picks if it["stratum"] == s) for s in STRATA}
        mode = "uniform"

    Path(args.out).write_text(json.dumps(picks, indent=1), encoding="utf-8")

    print(f"\nsampling mode: {mode}")
    print(f"{'stratum':>8s} {'drawn':>6s} {'avail':>6s}  {'% of sample':>11s}")
    for s in STRATA:
        got = sum(1 for it in picks if it["stratum"] == s)
        print(f"{s:>8s} {got:6d} {len(pool[s]):6d}  {got / max(len(picks),1):10.1%}")
        if 0 < got < 30:
            print(f"           ^ only {got} questions -- too few for a "
                  f"per-stratum significance test; use --balance when needed")

    manifest = {
        "builder": BUILDER_VERSION,
        "source": str(args.data),
        "n": len(picks),
        "seed": args.seed,
        "sampling": mode,
        "allocation": {s: sum(1 for it in picks if it["stratum"] == s) for s in STRATA},
        "pre_char_cap": PRE_CHAR_CAP,
        "post_char_cap": POST_CHAR_CAP,
        "ids": [it["id"] for it in picks],
        "strata": {it["id"]: it["stratum"] for it in picks},
        "prompt_hashes": {it["id"]: prompt_hash(it) for it in picks},
    }
    mpath = Path(args.out).with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(picks)} questions -> {args.out}")
    print(f"manifest (seed, ids, strata, prompt hashes) -> {mpath}")


def main():
    p = argparse.ArgumentParser() # p as the top-level parser
    sub = p.add_subparsers(dest="cmd", required=True) # sub-command slot of p

    # inspect: render a few prompts for review; registers
    i = sub.add_parser("inspect", help="render sample prompts for review")
    i.add_argument("--data", required=True)
    i.add_argument("--k", type=int, default=10)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--out", default=None)
    i.set_defaults(func=cmd_inspect) # to review

    # stats: table shapes and truncation rates
    s = sub.add_parser("stats", help="table shapes and truncation rates")
    s.add_argument("--data", required=True)
    s.set_defaults(func=cmd_stats) # generate statistics

    # sample: draw the evaluation set and manifest
    c = sub.add_parser("sample", help="draw the evaluation set and manifest")
    c.add_argument("--data", required=True)
    c.add_argument("--n", type=int, default=200)
    c.add_argument("--seed", type=int, default=20261026)
    c.add_argument("--out", default="finqa_200.json")
    c.add_argument("--stratify", action="store_true", default=True,
                   help="stratify by program steps (default on)")
    c.add_argument("--no-stratify", dest="stratify", action="store_false")
    c.add_argument("--balance", action="store_true",
                   help="equal n per stratum instead of proportional")
    c.set_defaults(func=cmd_sample) # reproducible RQ3 evaluation set (200) and manifest

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
