"""
RQ2 modification-set builder.

    python rq2_builder.py filter --data finqa_test.json
    python rq2_builder.py sample --data finqa_test.json --n 120 \
        --exclude finqa_rq3_200.manifest.json --out finqa_rq2_pool.json
    python rq2_builder.py build  --pool finqa_rq2_pool.json --keep 50 \
        --out rq2_out/modifications.jsonl
    python rq2_builder.py inspect --built rq2_out/modifications.jsonl --k 2

WHAT THIS DOES
--------------
FinQA supplies questions, a gold reasoning program, and gold supporting
facts -- but no responses. RQ2 needs responses, so this renders them:
a correct, well-written base answer built mechanically from the gold
program, then six damaged versions of it, one per criterion.

NO LANGUAGE MODEL IS INVOLVED. Base answers come from a template over
FinQA's own annotations; damage comes from seeded Python transforms.
Everything here is deterministic and reproducible from --seed. That is
the point: RQ2 needs to know EXACTLY what changed between two responses,
which is precisely what a generated modification could not guarantee.

WHY QUESTIONS ARE FILTERED
--------------------------
Not every FinQA item can carry six distinct damages. A single-step
question has no input to drop and no step order to shuffle. A question
whose figures come only from prose has no table row to cite, so the
provenance clause -- the thing the transparency damage removes -- has
nothing to name.

Filtering to multi-step, table-cited questions makes the RQ2 set
systematically harder and more structured than FinQA as a whole. That is
the right trade for an instrument test, but it MUST be stated in methods:
the RQ2 set is not a random sample of FinQA, and RQ3's 200 questions
(drawn from the full distribution) are not comparable to it.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

BUILDER_VERSION = "rq2_builder/v1"

CRITERIA = ["numerical_accuracy", "completeness", "clarity", "succinctness",
            "reasoning_transparency", "depth_appropriateness"]

TABLE_OPS = re.compile(r"table_(sum|average|max|min)")
NUM_ARG = re.compile(r"-?[\d.]+%?$")


# ------------------------------------------------------------------ filter

def unsuitable(item) -> list:
    """Reasons this item cannot carry six distinct modifications."""
    qa = item.get("qa") or item.get("qa_0") or {}
    why = []
    if not qa.get("ann_table_rows"):
        why.append("no table row cited")          # nothing to attribute to
    if len(qa.get("steps", [])) < 2:
        why.append("single step")                 # too thin to damage six ways
    if TABLE_OPS.search(qa.get("program", "")):
        why.append("table op")                    # not renderable as arithmetic
    if "greater" in qa.get("program", ""):
        why.append("yes/no answer")
    try:
        float(qa.get("exe_ans"))
    except (TypeError, ValueError):
        why.append("non-numeric answer")
    for s in qa.get("steps", []):
        for a in (str(s.get("arg1", "")), str(s.get("arg2", ""))):
            if not (a.startswith("#") or NUM_ARG.match(a.replace(",", ""))):
                why.append("unparseable argument")
                break
        else:
            continue
        break
    return why


def cmd_filter(args):
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    why = Counter()
    pool = []
    for it in data:
        reasons = unsuitable(it)
        for r in set(reasons):
            why[r] += 1
        if not reasons:
            pool.append(it)
    n = len(data)
    print(f"items                     : {n}")
    print(f"suitable for templating   : {len(pool)}  ({len(pool)/n:.1%})")
    print("\nexclusion reasons (overlapping):")
    for r, c in why.most_common():
        print(f"  {r:24s} {c:5d}  ({c/n:5.1%})")
    if args.out:
        Path(args.out).write_text(json.dumps(pool, indent=1))
        print(f"\npool -> {args.out}")
    return pool


def cmd_sample(args):
    pool = cmd_filter(argparse.Namespace(data=args.data, out=None))
    exclude = set()
    if args.exclude and Path(args.exclude).exists():
        exclude = set(json.loads(Path(args.exclude).read_text())["ids"])
        pool = [it for it in pool if str(it["id"]) not in exclude]
        print(f"\nafter excluding {len(exclude)} RQ3 ids: {len(pool)} available")
    rng = random.Random(args.seed)
    picks = rng.sample(pool, min(args.n, len(pool)))
    Path(args.out).write_text(json.dumps(picks, indent=1))
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps({
        "builder": BUILDER_VERSION, "source": str(args.data),
        "n": len(picks), "seed": args.seed,
        "excluded_from": str(args.exclude),
        "ids": [str(it["id"]) for it in picks],
    }, indent=2))
    print(f"\n{len(picks)} questions -> {args.out}")
    print("oversampled on purpose: `build --keep 50` discards any question "
          "where a modification cannot be applied cleanly")


# ------------------------------------------------------------------ render

def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(str(a).replace(",", "").rstrip("%"))
                   - float(str(b).replace(",", "").rstrip("%"))) < 1e-9
    except ValueError:
        return False


def provenance(value: str, gold_inds: dict):
    """
    Find where a figure came from, in two tiers.

    Tier 1 (labelled) -- FinQA renders table gold_inds as sentences, e.g.

        table_4: "( in millions ) the standby letters of credit of 2007
                  is 4711 ; the standby letters of credit of 2006 is 4926 ;"

    so the row label and period lift out directly.

    Tier 2 (contextual) -- text gold_inds are prose and often list several
    figures at once ("the average exercise price was $26.79, $33.32 and
    $26.93 for 2018, 2017 and 2016"). No per-figure label exists, so the
    sentence's leading descriptive clause is used instead. Less precise,
    but it still names a source the reader can check, which is what the
    transparency criterion requires.

    Returns (phrase, source_key, tier) or (None, None, None).
    """
    for key, sent in (gold_inds or {}).items():
        s = re.sub(r"\s+", " ", sent).strip()
        for m in re.finditer(r"the ([^;]{3,90}?) is \$?\s*(-?[\d.,]+)", s):
            if _num_eq(m.group(2), value):
                return re.sub(r"\s+", " ", m.group(1)).strip(), key, "labelled"

    for key, sent in (gold_inds or {}).items():
        s = re.sub(r"\s+", " ", sent).strip()
        for m in re.finditer(r"-?[\d.,]+", s):
            if not _num_eq(m.group(0), value):
                continue
            lead = re.sub(r"^\(\s*\d+\s*\)\s*", "", s)      # drop "( 1 )"
            lead = re.sub(r"^(the|a)\s+", "", lead)
            # cut at the verb: everything after it is the figure list itself
            lead = re.split(r"\b(?:is|was|were|are|of the)\b", lead)[0]
            words = lead.split()[:12]
            phrase = " ".join(words).strip(" ,;:.$")
            if len(phrase) < 3:
                continue
            return phrase, key, "context"
    return None, None, None


def units_of(gold_inds: dict) -> str:
    for sent in (gold_inds or {}).values():
        m = re.search(r"\(([^)]*(?:million|billion|thousand)[^)]*)\)", sent)
        if m:
            return m.group(1).strip()
    return ""


OP_WORDS = {"minus": ("subtract", "-"), "add": ("add", "+"),
            "divide": ("divide", "/"), "multiply": ("multiply", "*"),
            "subtract": ("subtract", "-")}


def op_of(step) -> tuple:
    base = re.sub(r"\d.*$", "", step.get("op", "")).strip("-_")
    return OP_WORDS.get(base, (base or "compute", "?"))


def render_base(item) -> dict:
    """
    Build a correct base answer. Shape (one clause per role, so each
    modification has a clean target):

        Sources: <figure> is <value> [<units>] (<row ref>); ...
        Step 1: <a> <op> <b> = <res>.
        Step 2: ...
        Answer: <exe_ans>

    Returns a dict of the PARTS as well as the text, because the damage
    transforms edit parts, not strings -- editing rendered prose with
    regexes is how modifications end up damaging more than intended.
    """
    qa = item.get("qa") or {}
    steps, gi = qa["steps"], qa.get("gold_inds", {})
    units = units_of(gi)

    sources, seen = [], set()
    untraced = []
    for s in steps:
        for a in (str(s["arg1"]), str(s["arg2"])):
            if a.startswith("#") or a in seen:
                continue
            phrase, key, tier = provenance(a, gi)
            seen.add(a)
            if phrase is None:
                # A figure the derivation uses but no gold_ind accounts for.
                # The BASE answer must be complete -- shipping one that
                # silently omits an input would mean the completeness
                # damage is already present before it is applied.
                untraced.append(a)
                continue
            row = key.replace("table_", "table row ").replace("text_", "text line ")
            sources.append({"value": a, "phrase": phrase, "row": row,
                            "tier": tier})

    lines = []
    for i, s in enumerate(steps, 1):
        word, sym = op_of(s)
        # FinQA numbers program results from #0; the rendered steps are
        # numbered from 1, so a reference to #0 is "step 1".
        a1 = f"the result of step {int(str(s['arg1'])[1:]) + 1}" \
            if str(s["arg1"]).startswith("#") else str(s["arg1"])
        a2 = f"the result of step {int(str(s['arg2'])[1:]) + 1}" \
            if str(s["arg2"]).startswith("#") else str(s["arg2"])
        lines.append({"n": i, "word": word, "sym": sym,
                      "a1": a1, "a2": a2, "res": str(s["res"])})

    # The displayed answer must agree with the last step. FinQA stores
    # exe_ans in raw form (0.33084) while the step result is often the
    # human-readable form (33.1%); showing one after the other would make
    # the BASE answer internally inconsistent -- a clarity failure present
    # before any damage is applied. exe_ans is kept for grading only.
    shown = str(lines[-1]["res"]) if lines else str(qa["exe_ans"])
    return {"sources": sources, "steps": lines, "untraced": untraced,
            "answer": shown, "exe_ans": str(qa["exe_ans"]), "units": units,
            "question": qa["question"]}


def to_text(parts) -> str:
    out = []
    if parts["sources"]:
        u = f" ({parts['units']})" if parts["units"] else ""
        src = "; ".join(f"{s['phrase']} = {s['value']}"
                        + (f" [{s['row']}]" if s.get("row") else "")
                        for s in parts["sources"])
        out.append(f"Sources{u}: {src}.")
    for s in parts["steps"]:
        sym = s["sym"]
        if sym == "-":
            clause = f"subtract {s['a2']} from {s['a1']}"
        elif sym == "+":
            clause = f"add {s['a1']} and {s['a2']}"
        elif sym == "/":
            clause = f"divide {s['a1']} by {s['a2']}"
        elif sym == "*":
            clause = f"multiply {s['a1']} by {s['a2']}"
        else:
            clause = f"{s['word']} {s['a1']} and {s['a2']}"
        out.append(f"Step {s['n']}: {clause}, giving {s['res']}.")
    if parts.get("prefix"):
        out.insert(0, parts["prefix"])
    if parts.get("suffix"):
        out.append(parts["suffix"])
    out.append(f"Answer: {parts['answer']}")
    return "\n".join(out)


# ------------------------------------------------------------------ damage

def _corrupt_figure(value: str, rng) -> str:
    """
    Introduce an arithmetic error in a figure. Prefers a transposition of
    two adjacent digits -- the classic slip -- and falls back to altering
    one digit where no such pair exists (e.g. "9.9%", whose digits are not
    adjacent in the string). Returns None if the value has no digits.
    """
    digits = [i for i, c in enumerate(value) if c.isdigit()]
    if not digits:
        return None
    pairs = [(a, b) for a, b in zip(digits, digits[1:])
             if b == a + 1 and value[a] != value[b]]
    if pairs:
        a, b = rng.choice(pairs)
        return value[:a] + value[b] + value[a] + value[b + 1:]
    i = rng.choice(digits)
    replacement = rng.choice([d for d in "0123456789" if d != value[i]])
    return value[:i] + replacement + value[i + 1:]


def mod_numerical_accuracy(p, rng):
    """Final figure wrong; the derivation above it no longer supports it."""
    new = _corrupt_figure(p["answer"], rng)
    if new is None or new == p["answer"]:
        return None
    q = json.loads(json.dumps(p))
    q["answer"] = new
    q["steps"][-1]["res"] = new
    return q


def mod_completeness(p, rng):
    """One required input never introduced; a step uses a figure the
    reader was never given."""
    if len(p["sources"]) < 2:
        return None
    q = json.loads(json.dumps(p))
    q["sources"].pop(rng.randrange(len(q["sources"])))
    return q


def mod_clarity(p, rng):
    """Steps out of order: a later step is presented before the step whose
    result it consumes, so the derivation cannot be followed in sequence."""
    if len(p["steps"]) < 2:
        return None
    q = json.loads(json.dumps(p))
    q["steps"] = list(reversed(q["steps"]))
    return q


PAD_PREFIX = ("Before answering, it is worth restating the question: "
              "{question} This is a common form of financial question and "
              "the approach below follows the standard method.")
PAD_SUFFIX = ("Please note that these figures are drawn from the document "
              "provided and have not been independently verified. Results "
              "of this kind should be interpreted with care, and readers "
              "may wish to consult the full filing for further context. "
              "Past performance is not necessarily indicative of future "
              "results.")


def mod_succinctness(p, rng):
    """Material beyond what the question requires: a restatement and
    unrequested caveats. Content and derivation untouched."""
    q = json.loads(json.dumps(p))
    q["prefix"] = PAD_PREFIX.format(question=p["question"])
    q["suffix"] = PAD_SUFFIX
    return q


def mod_reasoning_transparency(p, rng):
    """Inputs and operations no longer named: the figure is right, but the
    derivation cannot be reconstructed or checked."""
    q = json.loads(json.dumps(p))
    q["sources"], q["steps"] = [], []
    q["prefix"] = ("Based on the data provided, the required figure "
                   "follows directly from the relevant values.")
    return q


DEPTH_SUFFIX = ("It is worth extending this analysis further. Applying a "
                "discounted cash flow treatment to the same figures, and "
                "assuming a constant discount rate across the period, the "
                "implied present value would shift materially under a "
                "range of terminal growth assumptions; a Monte Carlo "
                "simulation over those assumptions would place the "
                "interquartile outcome across a wide band. A sensitivity "
                "matrix over discount rate and growth rate would refine "
                "this further, as would a comparison against sector peers "
                "on an EV/EBITDA basis.")


def mod_depth_appropriateness(p, rng):
    """Depth mismatched to the question: a multi-method valuation
    discussion where a single ratio was asked for."""
    q = json.loads(json.dumps(p))
    q["suffix"] = DEPTH_SUFFIX
    return q


def mod_misattribution(p, rng):
    """
    OPTIONAL SEVENTH. Figures correct, provenance false: the value is
    cited to a row it did not come from, so a reviewer checking the cited
    row finds a different number.

    WHICH CRITERION THIS TARGETS IS A DESIGN DECISION, NOT A FACT. The
    arithmetic is right, so it is not an accuracy failure under the
    locked definition; but the audit trail is broken, which is what
    transparency measures. Assigned to transparency here. If you include
    it, say so in methods and give the reasoning -- do not leave it
    implicit.
    """
    if not p["sources"]:
        return None
    q = json.loads(json.dumps(p))
    i = rng.randrange(len(q["sources"]))
    src = q["sources"][i]
    m = re.search(r"(\d{4})", src["phrase"])
    if m:
        src["phrase"] = src["phrase"].replace(
            m.group(1), str(int(m.group(1)) + rng.choice([-2, -1, 1])))
    elif src.get("row"):
        m2 = re.search(r"(\d+)$", src["row"])
        if not m2:
            return None
        src["row"] = src["row"][:m2.start()] + str(int(m2.group(1)) + 1)
    else:
        return None
    return q


MODIFICATIONS = {
    "corrupt_figure": ("numerical_accuracy", mod_numerical_accuracy),
    "drop_input": ("completeness", mod_completeness),
    "reverse_steps": ("clarity", mod_clarity),
    "add_padding": ("succinctness", mod_succinctness),
    "remove_derivation": ("reasoning_transparency", mod_reasoning_transparency),
    "append_overdepth": ("depth_appropriateness", mod_depth_appropriateness),
}
OPTIONAL = {"misattribution": ("reasoning_transparency", mod_misattribution)}


# ------------------------------------------------------------------- build

def cmd_build(args):
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    mods = dict(MODIFICATIONS)
    if args.include_misattribution:
        mods.update(OPTIONAL)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, skipped = [], Counter()
    records = []

    for item in pool:
        if len(kept) >= args.keep:
            break
        parts = render_base(item)
        if not parts["sources"]:
            skipped["no figure could be traced to a source"] += 1
            continue
        if parts["untraced"]:
            # base would use figures it never introduces -- i.e. it would
            # already fail completeness before any damage is applied
            skipped["base incomplete: untraced figure"] += 1
            continue
        variants = {}
        fail = None
        for name, (target, fn) in mods.items():
            q = fn(parts, rng)
            if q is None:
                fail = name
                break
            variants[name] = (target, to_text(q))
        if fail:
            skipped[f"{fail} not applicable"] += 1
            continue

        qid = str(item["id"])
        kept.append(qid)
        records.append({"qid": qid, "variant": "base", "target_criterion": None,
                        "question": parts["question"], "gold": parts["exe_ans"],
                        "response": to_text(parts)})
        for name, (target, text) in variants.items():
            records.append({"qid": qid, "variant": name,
                            "target_criterion": target,
                            "question": parts["question"],
                            "gold": parts["answer"], "response": text})

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "builder": BUILDER_VERSION, "pool": str(args.pool), "seed": args.seed,
        "questions_kept": len(kept), "questions_requested": args.keep,
        "modifications": {k: v[0] for k, v in mods.items()},
        "records": len(records), "ids": kept,
        "skipped": dict(skipped),
        "no_language_model": True,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"questions kept   : {len(kept)} / {args.keep} requested")
    print(f"records written  : {len(records)} "
          f"({len(kept)} base + {len(kept) * len(mods)} modified)")
    if skipped:
        print("skipped:")
        for k, v in skipped.most_common():
            print(f"  {k:40s} {v}")
    print(f"\n-> {out_path}")
    print(f"-> {out_path.with_suffix('.manifest.json')}")
    if len(kept) < args.keep:
        print("\nWARNING: fewer questions than requested. Sample a larger "
              "pool and rerun.")
    print("\nNEXT: finalise the expectation table BEFORE scoring these.")


def cmd_inspect(args):
    recs = [json.loads(l) for l in open(args.built, encoding="utf-8")]
    by_q = {}
    for r in recs:
        by_q.setdefault(r["qid"], []).append(r)
    for qid in list(by_q)[:args.k]:
        print("=" * 74)
        print(f"{qid}\nQ: {by_q[qid][0]['question']}")
        for r in by_q[qid]:
            tag = r["variant"] if r["variant"] == "base" else \
                f"{r['variant']} -> {r['target_criterion']}"
            print("-" * 74)
            print(f"[{tag}]")
            print(r["response"])
        print()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("filter", help="report how many items are templatable")
    f.add_argument("--data", required=True)
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_filter)

    s = sub.add_parser("sample", help="draw an oversampled pool")
    s.add_argument("--data", required=True)
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=20261026)
    s.add_argument("--exclude", default="finqa_rq3_200.manifest.json")
    s.add_argument("--out", default="finqa_rq2_pool.json")
    s.set_defaults(func=cmd_sample)

    b = sub.add_parser("build", help="render base answers + six modifications")
    b.add_argument("--pool", default="finqa_rq2_pool.json")
    b.add_argument("--keep", type=int, default=50)
    b.add_argument("--seed", type=int, default=20261026)
    b.add_argument("--include-misattribution", action="store_true")
    b.add_argument("--out", default="rq2_out/modifications.jsonl")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("inspect", help="print built variants for eyeballing")
    i.add_argument("--built", default="rq2_out/modifications.jsonl")
    i.add_argument("--k", type=int, default=2)
    i.set_defaults(func=cmd_inspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()