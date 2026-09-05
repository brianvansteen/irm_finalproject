"""
RUN SCRIPT 1 of 5: setup (Phases 0 and 1).

    python run_1_setup.py               # do everything, skipping finished steps
    python run_1_setup.py --dry-run     # show what would run, run nothing
    python run_1_setup.py --force       # redo even finished steps
    python run_1_setup.py --allow-cpu   # proceed WITHOUT a GPU (see below)

Runs, in order:
  1. environment check (Python libraries, versions, GPU actually usable)
  2. UltraFeedback diagnostics        -> logs/diagnostics_ultrafeedback.txt
  3. FinQA prompt statistics          -> logs/stats_finqa.txt
  4. criteria build + audit           -> criteria.npz (+ manifest)
  5. embedding caches, SIX runs       -> emb_<condition>_<split>.npz
     (3 conditions x train/test; every condition needs both splits,
      because each condition's W_Q is retrained on that encoder's
      own training embeddings)
  6. disjoint question samples for RQ3 first (200) and then RQ2 (50)

Every step is skipped automatically if its output file already exists,
so the script can be re-run safely after an interruption.

Requires: finqa_test.json (from the FinQA repository).

THE CPU GATE
-----------------------------------------------------
Step 5 is the time-consumping one. Six encoder passes over ~60k pairs each.
On a GPU that is minutes; on CPU it is hours, and the pipeline does not
discriminate since every module selects its device with

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

Pass --allow-cpu to run without a GPU.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from load_ultrafeedback import ENCODER, ENCODER_REVISION, MAX_LEN

# ------------------------------------------------------------ conditions
# The three RQ1 conditions. Names appear in cache filenames, run folders,
# and the final results table as one source of truth, imported by run 2.

DISTILBERT = "distilbert-base-uncased"
DISTILBERT_REVISION = "12040ac"   # head of main; repo dormant >1 year

CONDITIONS = {
    "modernbert_full": {"encoder": ENCODER, "revision": ENCODER_REVISION,
                        "max_len": MAX_LEN},
    "modernbert_512":  {"encoder": ENCODER, "revision": ENCODER_REVISION,
                        "max_len": 512},
    "distilbert_512":  {"encoder": DISTILBERT,
                        "revision": DISTILBERT_REVISION, "max_len": 512},
}
MAIN_CONDITION = "modernbert_full" # the reported system; RQ2/RQ3 use this
SPLITS = ("train_prefs", "test_prefs")

FINQA_TEST = "finqa_test.json"
RQ3_N, RQ3_SEED = 200, 20261026

# RQ2 draws from a filtered pool (multi-step, table-cited, templatable --
# roughly 220 of the 1,147 test items), oversampled because some questions
# lose a modification during rendering. The draw itself lives in
# rq2_builder.py; this script sequences it and verifies disjointness.
RQ2_POOL_N, RQ2_KEEP, RQ2_SEED = 120, 50, 20261027

# Sampling modes with FinQA's natural difficulty mix is roughly 59% one-step,
# 33% two-step, 8% three-plus.
# PROPORTIONAL preserves that mix, so the headline RQ3 accuracy is representative
# of the dataset.
# The alternative (--balance, equal counts per stratum) buys enough
# questions for per-stratum significance tests but makes the pooled
# accuracy unrepresentative, so it must then be reported per statum.

STRATIFY_MODE = "proportional"

LOGS = Path("logs")


def cache_path(condition: str, split: str) -> Path:
    short = {"train_prefs": "train", "test_prefs": "test"}[split]
    return Path(f"emb_{condition}_{short}.npz")


# ------------------------------------------------------------ machinery

def run(cmd, log_to=None, dry=False):
    str_cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(str_cmd)}")
    if dry:
        return ""
    res = subprocess.run([sys.executable] + str_cmd,
                         capture_output=bool(log_to), text=True)
    if log_to and res.stdout:
        LOGS.mkdir(exist_ok=True)
        Path(log_to).write_text(res.stdout)
        print(res.stdout[-1500:])
        print(f"  (full output saved to {log_to})")
    if res.returncode != 0:
        if log_to and res.stderr:
            print(res.stderr[-2000:])
        raise SystemExit(f"step failed: {' '.join(map(str, cmd))}")
    return res.stdout or ""


def step(name, output, force):
    """Returns True if the step should run (output missing or --force)."""
    if output and Path(output).exists() and not force:
        print(f"[skip] {name}: {output} already exists")
        return False
    print(f"[run ] {name}")
    return True


# ---------------------------------------------------------------- steps

def check_environment(dry, allow_cpu=False):
    print("[run ] environment check")
    if dry:
        return
    import torch
    import transformers

    # transformers 4.48.0 required
    from check_env import MIN_TRANSFORMERS, _ver_tuple

    tv = transformers.__version__
    tf_ok = _ver_tuple(tv) >= MIN_TRANSFORMERS
    want = ".".join(str(p) for p in MIN_TRANSFORMERS)

    cuda_build = torch.version.cuda is not None
    cuda_live = bool(torch.cuda.is_available())

    print(f"  python       {sys.version.split()[0]}")
    print(f"  torch        {torch.__version__}")
    print(f"  cuda build   {torch.version.cuda or 'NONE -- CPU-ONLY WHEEL'}")
    print(f"  transformers {tv} "
          f"{'OK' if tf_ok else f'<-- TOO OLD, need >= {want} for ModernBERT'}")
    if cuda_live:
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA available: {torch.cuda.get_device_name(0)} "
              f"(sm_{cap[0]}{cap[1]})")
    else:
        print("CUDA NOT available")

    if not tf_ok:
        raise SystemExit(
            f"transformers {tv} is too old; ModernBERT needs >= {want}. "
            f"pip install -U 'transformers>={want}'")

    # ---- the CPU gate --------------------------------------------------
    if not (cuda_build and cuda_live) and not allow_cpu:
        why = ("torch has NO CUDA build -- this is the CPU-only wheel "
               f"({torch.__version__})" if not cuda_build else
               "torch is a CUDA build but no device is visible")
        raise SystemExit(
            f"\nREFUSING TO START: {why}.\n\n"
            f"Step 5 of this script is six encoder passes over ~60k pairs "
            f"each.\nOn a GPU that is minutes; on CPU it is hours.")

    if allow_cpu and not cuda_live:
        print("\n  *** --allow-cpu: proceeding WITHOUT a GPU. Embedding will "
              "take hours. ***\n")


def finqa_samples(args):
    """
    RQ3 first, then RQ2 excluding the RQ3 samples chosen.

    The RQ2 draw is delegated to `rq2_builder.py sample`, which filters for
    templatable questions before sampling and takes --exclude.
    """
    if step("sample RQ3 questions", "finqa_rq3_200.json", args.force):
        cmd = ["finqa_prompt.py", "sample", "--data", FINQA_TEST,
               "--n", RQ3_N, "--seed", RQ3_SEED,
               "--stratify", "--out", "finqa_rq3_200.json"]
        if STRATIFY_MODE == "balanced":
            cmd.append("--balance")
        run(cmd, dry=args.dry_run)

    if step("sample RQ2 pool (excludes RQ3 ids)", "finqa_rq2_pool.json",
            args.force):
        run(["rq2_builder.py", "sample", "--data", FINQA_TEST,
             "--n", RQ2_POOL_N, "--seed", RQ2_SEED,
             "--exclude", "finqa_rq3_200.manifest.json",
             "--out", "finqa_rq2_pool.json"],
            log_to=LOGS / "sample_rq2.txt", dry=args.dry_run)

    if step("build RQ2 modification set", "rq2_out/modifications.jsonl",
            args.force):
        run(["rq2_builder.py", "build", "--pool", "finqa_rq2_pool.json",
             "--keep", RQ2_KEEP, "--seed", RQ2_SEED,
             "--out", "rq2_out/modifications.jsonl"],
            log_to=LOGS / "build_rq2.txt", dry=args.dry_run)

    if args.dry_run:
        return

    # Verify disjoint samples. That exclusion happens inside rq2_builder,
    # so this is an independent check that it worked.
    rq3 = set(json.loads(
        Path("finqa_rq3_200.manifest.json").read_text())["ids"])
    rq2 = set(json.loads(
        Path("rq2_out/modifications.manifest.json").read_text())["ids"])
    overlap = rq3 & rq2
    if overlap:
        raise SystemExit(
            f"RQ2 and RQ3 question sets overlap on {len(overlap)} ids "
            f"({sorted(overlap)[:3]}...). Do not proceed: the two sets must "
            f"be disjoint or RQ2's modifications leak into RQ3 scoring.")
    print(f"\ndisjointness verified: RQ3 {len(rq3)} questions, "
          f"RQ2 {len(rq2)} questions, 0 shared")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="proceed without a GPU. Embedding will take hours.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="embed.py batch size; reduce if limited "
                         "GPU memory for long max-len")
    args = ap.parse_args()

    if not Path(FINQA_TEST).exists() and not args.dry_run:
        raise SystemExit(
            f"{FINQA_TEST} not found. Download the FinQA test split from "
            f"the FinQA repository and place it here first.")

    check_environment(args.dry_run, allow_cpu=args.allow_cpu)

    if step("UltraFeedback load + diagnostics",
            LOGS / "diagnostics_ultrafeedback.txt", args.force):
        run(["load_ultrafeedback.py"],
            log_to=LOGS / "diagnostics_ultrafeedback.txt", dry=args.dry_run)

    if step("FinQA prompt statistics", LOGS / "stats_finqa.txt", args.force):
        run(["finqa_prompt.py", "stats", "--data", FINQA_TEST],
            log_to=LOGS / "stats_finqa.txt", dry=args.dry_run)

    print("\n*** READ logs/diagnostics_ultrafeedback.txt and "
          "logs/stats_finqa.txt NOW. ***\nMAX_LEN in load_ultrafeedback.py "
          f"is {MAX_LEN}; confirm the truncation tables support it before "
          "embedding. If you change MAX_LEN, rerun this script.")

    if step("criteria build", "criteria.npz", args.force):
        run(["criteria_builder.py", "build", "--out", "criteria.npz"],
            dry=args.dry_run)
        run(["criteria_builder.py", "audit", "--criteria", "criteria.npz"],
            log_to=LOGS / "criteria_audit.txt", dry=args.dry_run)

    for cond, cfg in CONDITIONS.items():
        for split in SPLITS:
            out = cache_path(cond, split)
            if step(f"embed {cond} / {split}", out, args.force):
                run(["embed.py", "--split", split, "--out", out,
                     "--encoder", cfg["encoder"],
                     "--encoder-revision", cfg["revision"],
                     "--max-len", cfg["max_len"],
                     "--batch-size", args.batch_size], dry=args.dry_run)

    finqa_samples(args)

    print("\nSetup complete.")
    print("Review rq2_out/modifications.jsonl")


if __name__ == "__main__":
    main()
