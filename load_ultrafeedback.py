"""
Loader + diagnostics for allenai/ultrafeedback_binarized_cleaned.

  - Same split names/sizes as HuggingFaceH4/ultrafeedback_binarized
    (train_prefs ~60.8k, test_prefs ~1.96k), so it is a drop-in swap.
  - Faulty rows identified by Argilla (high overall_score, damning critique)
    have been removed.
  - TruthfulQA prompts removed, so no benchmark contamination.
  - Adds a `source` column to filter/stratify by origin.

Reproducibility:
https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned/commits/main

"""

from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "allenai/ultrafeedback_binarized_cleaned"
# As of August 2026: repo has been domnant since January 12, 2024.
# "This is a version of the UltraFeedback binarized dataset but
# with TruthfulQA prompts removed and source annotations added
# (so you can filter out samples from different sources yourself if you want!)."
# `revision=f304ce5`
REVISION = "f304ce5"

# Primary system. ModernBERT supports 8192 natively -- no position surgery.
# Pin the revision hash.
ENCODER = "answerdotai/ModernBERT-base"

# Head of main as of 2026-09-01: "Set tokenizer model_max_length property
# to 8192 (#39)", 2025-01-15. Repo effectively dormant since.
# Requires transformers >= 4.48 (ModernBERT architecture added in v4.48.0).
ENCODER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
# ENCODER_REVISION = "890363e1dc0f82175ed809ed6635d549acd1f1f9"

# RQ1 ablation conditions (encoder, max_len):
#   ("distilbert-base-uncased",     512)
#   ("answerdotai/ModernBERT-base", 512)
#   ("answerdotai/ModernBERT-base", MAX_LEN)
#
# ModernBERT supports 8192 max_position_embeddings:
#
#   UltraFeedback  prompt ~185 + response ~305 tokens  -> ~490 token typical
#   FinQA          input avg 687, max 2,679 (Chen et al. 2022, Table 1)
#                  + response, so ~1,200 typical / ~3,200 worst case
#
# One max_length must serve both: RQ1 trains on UltraFeedback, RQ3 scores FinQA
# with the same checkpoint. Training short and scoring long means asking the
# model to extrapolate to sequence lengths it never saw.

MAX_LEN = 2048


# ---------------------------------------------------------------- extraction

def extract_pair(row):
    """
    `chosen` and `rejected` are *full conversations*, not strings.
    Each is a list:
        [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    The user turn is identical in both and equals `prompt`.

    `score_chosen` / `score_rejected` are the 0-10 overall_score.
    """
    return {
        "prompt": row["prompt"],
        "chosen_text": row["chosen"][-1]["content"],
        "rejected_text": row["rejected"][-1]["content"],
        "score_chosen": row["score_chosen"],
        "score_rejected": row["score_rejected"],
        "source": row["source"],
    }


def is_usable(row):
    """
    Two filters that matter:

    1. Ties. Some rows have score_chosen == score_rejected (e.g. 9.0 vs 9.0,
       3.0 vs 3.0).
       Removing them is standard and I will report how many were dropped.

    2. Empty or whitespace-only responses.
    """
    if row["score_chosen"] is None or row["score_rejected"] is None:
        return False
    if row["score_chosen"] <= row["score_rejected"]:
        return False
    if not row["chosen_text"].strip() or not row["rejected_text"].strip():
        return False
    return True


def load_prefs(split, revision=REVISION, drop_ties=True):
    ds = load_dataset(DATASET, split=split, revision=revision)
    before = len(ds)
    ds = ds.map(extract_pair, remove_columns=ds.column_names)
    if drop_ties:
        ds = ds.filter(is_usable)
    print(f"{split}: {before} -> {len(ds)} rows ({before - len(ds)} dropped)")
    return ds


# ------------------------------------------------------------- tokenisation

def make_tokenize_fn(tokenizer, max_len=MAX_LEN):
    """
    Encode (prompt, response) as a sentence pair so the encoder sees the
    separator token between instruction and answer. Produces two parallel
    encodings per row (chosen and rejected), which is what the Bradley-Terry
    style ranking loss calculates.
    """

    def fn(batch):
        chosen = tokenizer(
            batch["prompt"], batch["chosen_text"],
            truncation=True, max_length=max_len, padding=False,
        )
        rejected = tokenizer(
            batch["prompt"], batch["rejected_text"],
            truncation=True, max_length=max_len, padding=False,
        )
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    return fn


# --------------------------------------------------------------- diagnostics

def diagnostics(ds, tokenizer, sample=4000):
    """
    The headline number is `chosen longer than rejected`. As the winning
    response is the longer one, length bias is baked in training labels.
    That is (a) a limitation to state, and (b) the baseline RQ3 best-of-N
    length analysis has to be read against.

    The truncation rate details how much of the data a 512-token window is
    actually throwing away, relevant to the RQ1 ablation analysis.
    """
    n = min(sample, len(ds))
    sub = ds.select(range(n))

    longer = 0
    lens_c, lens_r = [], []

    for row in sub:
        tc = tokenizer(row["prompt"], row["chosen_text"], truncation=False)["input_ids"]
        tr = tokenizer(row["prompt"], row["rejected_text"], truncation=False)["input_ids"]
        lens_c.append(len(tc))
        lens_r.append(len(tr))
        if len(tc) > len(tr):
            longer += 1

    alllens = sorted(lens_c + lens_r)
    m = len(alllens)

    print(f"\n--- diagnostics on {n} pairs ---")
    print(f"chosen longer than rejected : {longer / n:6.1%}   <-- label length bias")
    print(f"mean tokens, chosen         : {sum(lens_c) / n:6.1f}")
    print(f"mean tokens, rejected       : {sum(lens_r) / n:6.1f}")
    print(f"median / p95 / p99 / max    : {alllens[m//2]} / {alllens[int(m*.95)]} "
          f"/ {alllens[int(m*.99)]} / {alllens[-1]}")

    # Truncation at each candidate sequence length, so MAX_LEN is chosen from evidence.
    # Compare against `finqa_prompt.py stats` before fixing.
    # One value has to serve UltraFeedback training AND FinQA scoring.
    print("\ntruncation rate by sequence length:")
    for w in (512, 1024, 2048, 4096, 8192):
        rate = sum(1 for L in alllens if L > w) / m
        flag = "  <-- current MAX_LEN" if w == MAX_LEN else ""
        print(f"  {w:5d}: {rate:6.2%}{flag}")
    print("pick the smallest sequence length that also covers FinQA "
          "(avg 687, max 2,679 + response)")

    print("\nsource breakdown:")
    for src, count in Counter(sub["source"]).most_common():
        print(f"  {src:24s} {count:6d}  ({count / n:5.1%})")


# --------------------------------------------------------------------- main

if __name__ == "__main__":
    import transformers
    print(f"transformers {transformers.__version__} (ModernBERT needs >= 4.48)")
    print(f"dataset: {DATASET} pin {REVISION}")
    print(f"encoder: {ENCODER} pin {ENCODER_REVISION[:20]}")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)

    train = load_prefs("train_prefs")
    test = load_prefs("test_prefs")

    diagnostics(train, tokenizer)

    tok_fn = make_tokenize_fn(tokenizer)
    train_tok = train.map(tok_fn, batched=True)
    test_tok = test.map(tok_fn, batched=True)

    print(f"\ntokenised: train={len(train_tok)} // test={len(test_tok)}")
    print("columns:", train_tok.column_names)