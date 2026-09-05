"""
Embed UltraFeedback pairs with the frozen encoder. Run ONCE per
(encoder, max_len) condition; everything downstream reads the cache.

    python embed.py --split train_prefs --out emb_train.npz
    python embed.py --split test_prefs  --out emb_test.npz

    # RQ1 ablation conditions (pin the revision, as run_1_setup.py does):
    python embed.py --split test_prefs --encoder distilbert-base-uncased \
        --encoder-revision 12040ac --max-len 512 --out emb_distilbert_512_test.npz
    python embed.py --split test_prefs --max-len 512 --out emb_modernbert_512_test.npz

Cache naming discipline: encode the condition in the filename. The sidecar
JSON records the exact provenance either way, but a filename that says what
it is prevents the wrong cache reaching the wrong run.

POOLING is a pinned decision, stated once in methods and identical across
all encoders in the ablation: CLS token (first position), for continuity
with the prior DistilBERT+HRR system (Van Steen, 2026). Both DistilBERT
and ModernBERT place CLS at position 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from load_ultrafeedback import (DATASET, ENCODER, ENCODER_REVISION, MAX_LEN,
                                REVISION, load_prefs)

POOLING = "cls"  # pinned; see module docstring


@torch.no_grad()
def embed_texts(prompts, texts, tokenizer, model, device, max_len, batch_size):
    out = []
    n = len(texts)
    use_bf16 = device.type == "cuda"
    n_batches = (n + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, n, batch_size)):
        enc = tokenizer(prompts[i:i + batch_size], texts[i:i + batch_size],
                        truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            h = model(**enc).last_hidden_state
        out.append(h[:, 0, :].float().cpu().numpy())  # CLS pooling
        done = min(i + batch_size, n)
        if b % 20 == 0 or done == n:
            print(f"  {done}/{n}  (batch {b + 1}/{n_batches})", flush=True)
    return np.vstack(out).astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    choices=["train_prefs", "test_prefs"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--encoder-revision", default=ENCODER_REVISION)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="drop this if you OOM at long max-len; 8-16 at 2048")
    ap.add_argument("--force", action="store_true",
                    help="re-embed even if --out already exists")
    args = ap.parse_args()

    if Path(args.out).exists() and not args.force:
        raise SystemExit(f"{args.out} already exists; pass --force to "
                         f"overwrite (this is a multi-minute encoder pass).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"encoder={args.encoder} @ {args.encoder_revision}")
    print(f"max_len={args.max_len}  pooling={POOLING}  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder, revision=args.encoder_revision)
    model = AutoModel.from_pretrained(
        args.encoder, revision=args.encoder_revision).to(device).eval()

    ds = load_prefs(args.split)
    prompts = ds["prompt"]
    print(f"\nembedding chosen ({len(prompts)} texts)...")
    emb_c = embed_texts(prompts, ds["chosen_text"], tokenizer, model,
                        device, args.max_len, args.batch_size)
    print(f"embedding rejected ({len(prompts)} texts)...")
    emb_r = embed_texts(prompts, ds["rejected_text"], tokenizer, model,
                        device, args.max_len, args.batch_size)

    np.savez_compressed(args.out, emb_chosen=emb_c, emb_rejected=emb_r)
    sidecar = {
        "dataset": DATASET, "dataset_revision": REVISION,
        "split": args.split, "n_pairs": int(emb_c.shape[0]),
        "encoder": args.encoder, "encoder_revision": args.encoder_revision,
        "max_len": args.max_len, "pooling": POOLING,
        "torch": torch.__version__,
    }
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2))
    print(f"\n{emb_c.shape} x2 -> {args.out}")
    print(f"provenance -> {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
