"""
HRR criteria builder. Constructs the SIX FROZEN stored patterns for the
Hopfield energy head, from the locked criterion definitions.

    python criteria_builder.py build   --out criteria.npz
    python criteria_builder.py audit   --criteria criteria.npz

CONSTRUCTION
------------
Vocabulary atoms are random unit HRR vectors (Plate 1995; implementation:
Kelly & Tomkins-Flanagan, ecphory/hrr), generated under one fixed seed.
Each criterion k is

    xi_k = unit( name_k  +  sum_j  name_k (*) concept_{k,j} )

where (*) is circular-convolution binding. The name term gives each
criterion a distinct identity (random names are near-orthogonal in 768-d);
the bound terms give it compositional content.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
The atoms are SYMBOLS: random vectors carrying labels, not word embeddings.
The atom "numerical_accuracy" is a name only; the vector knows nothing of
numbers or accuracy. The construction is stipulative -- the report says so
plainly. What HRR buys is not lexical semantics but AUDITABILITY: the
contents of each criterion can be verified after the fact by unbinding,

    xi_k (/) name_k  ~  sum_j concept_{k,j} + noise,

and checking which vocabulary atoms the result matches. The `audit` command
performs exactly that check and prints the retrieval table: every criterion
must recover ITS OWN concepts (hits) and none of the other criteria's
(false positives). That table is the appendix artefact: a demonstration
that the stored patterns contain what the documentation says they contain,
which is the SR 11-7 effective-challenge property in miniature.

Semantics enter the system elsewhere: W_Q learns, from preference data
alone, the translation from encoder space into this criteria space. The
division of labour -- stipulated structure here, learned meaning there --
is the architecture's central design choice.

SEPARATION
----------
Random unit vectors in 768-d have cosine ~ N(0, 1/768), sd ~ 0.036. The
6x6 off-diagonal similarities should sit within a few sd of zero; `build`
requires max |off-diagonal| < MAX_OFFDIAG (0.15) and prints the matrix
for the report. Near-orthogonal stored patterns matter downstream: they
are what makes the softmax attribution p read as "which criterion", rather
than as weight smeared across overlapping patterns.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from hrr import HRR

BUILDER_VERSION = "criteria_builder/v1"
DIM = 768
SEED = 42
MAX_OFFDIAG = 0.15

# ---------------------------------------------------------------- criteria
# Terms follow the locked definitions sheet ("Reward criteria: locked
# definitions"). Order fixes the row order of Xi and therefore the index
# meaning of the attribution vector p everywhere downstream. DO NOT REORDER.

CRITERIA = {
    "numerical_accuracy": [
        # figures arithmetically correct, correctly derived from source data
        "correct_figures", "correct_arithmetic", "correct_derivation", "from_source_data",
    ],
    "completeness": [
        # uses all information required, omits nothing pertinent
        "required_information", "all_inputs", "nothing_omitted", "pertinent",
    ],
    "clarity": [
        # internally consistent, logically ordered, unambiguous
        "consistent", "logical_order", "unambiguous", "followable",
    ],
    "succinctness": [
        # no material beyond what the question requires (NOT brevity)
        "no_superfluous", "only_requested", "no_padding", "no_restating",
    ],
    "reasoning_transparency": [
        # names its inputs and operations; derivation reconstructable
        "inputs_named", "operations_stated", "reconstructable", "checkable",
    ],
    "depth_appropriateness": [
        # analytical depth matches what the question requires (fit, not level)
        "depth_fit", "matches_question", "no_overreach", "no_underreach",
    ],
}


# ------------------------------------------------------------ construction

def make_vocabulary(seed: int = SEED, dim: int = DIM) -> dict:
    """One seeded pass generates every atom: 6 names + all concept terms.
    Generation order is fixed by the dict above, so the same seed always
    yields the same vectors."""
    state = np.random.get_state()
    np.random.seed(seed)
    vocab = {}
    for crit in CRITERIA:
        vocab[f"NAME:{crit}"] = HRR(N=dim)
    for crit, terms in CRITERIA.items():
        for t in terms:
            key = f"TERM:{t}"
            if key not in vocab:
                vocab[key] = HRR(N=dim)
    np.random.set_state(state)
    return vocab


def build_criteria(vocab: dict) -> np.ndarray:
    """xi_k = unit(name_k + sum_j name_k * concept_j), rows in CRITERIA order."""
    Xi = np.zeros((len(CRITERIA), DIM), dtype=np.float32)
    for k, (crit, terms) in enumerate(CRITERIA.items()):
        name = vocab[f"NAME:{crit}"]
        trace = HRR(data=name.v.copy())
        for t in terms:
            trace = trace + (name * vocab[f"TERM:{t}"])
        Xi[k] = (trace.v / np.linalg.norm(trace.v)).astype(np.float32)
    return Xi


def similarity_matrix(Xi: np.ndarray) -> np.ndarray:
    return Xi @ Xi.T  # rows are unit-norm


def unbinding_audit(Xi: np.ndarray, vocab: dict) -> list:
    """
    For each criterion row, unbind with its name and rank every TERM atom
    by cosine similarity to the residue. Records, per criterion:
      - similarity of each of its OWN terms (should be clearly positive)
      - the best-scoring FOREIGN term (should sit at noise level)
    """
    names = list(CRITERIA)
    term_keys = sorted(k for k in vocab if k.startswith("TERM:"))
    report = []
    for k, crit in enumerate(names):
        name = vocab[f"NAME:{crit}"]
        residue = HRR(data=Xi[k]) / name        # unbind
        rv = residue.v / np.linalg.norm(residue.v)
        sims = {t[5:]: float(rv @ (vocab[t].v / np.linalg.norm(vocab[t].v)))
                for t in term_keys}
        own = {t: sims[t] for t in CRITERIA[crit]}
        foreign = {t: s for t, s in sims.items() if t not in CRITERIA[crit]}
        top_foreign = max(foreign, key=foreign.get)
        min_own = min(own.values())
        top_foreign_sim = foreign[top_foreign]
        report.append({
            "criterion": crit,
            "own_terms": {t: round(s, 3) for t, s in own.items()},
            "min_own": round(min_own, 3),
            "top_foreign_term": top_foreign,
            "top_foreign_sim": round(top_foreign_sim, 3),
            "separated": min_own > top_foreign_sim,
        })
    return report


# ------------------------------------------------------------------ commands

def similarity_figure(S, names, out="fig_criteria_similarity"):
    """
    The 6x6 cosine similarity matrix as a figure for Design and Methodology.

    It makes one claim visually that costs a paragraph in prose: the six
    stored patterns are near-orthogonal by construction, so the softmax
    attribution reads as "which criterion" rather than as weight smeared
    across overlapping directions. The dashed reference in the caption is
    1/sqrt(d), the standard deviation of cosine similarity between random
    unit vectors at this dimensionality -- off-diagonal cells should sit
    within a few multiples of it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")
        return
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.1})

    off = np.abs(S - np.eye(len(names)))
    lim = max(off.max(), 1e-6)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    # scale to the OFF-diagonal range: on a -1..1 scale the diagonal
    # dominates and every off-diagonal cell looks identically white
    im = ax.imshow(S, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", " ") for n in names],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, "1.00" if i == j else f"{S[i, j]:+.3f}",
                    ha="center", va="center", fontsize=7.5,
                    color="black" if i != j else "white")
    ax.set_title(f"Criterion vector cosine similarity\n"
                 f"max |off-diagonal| = {off.max():.3f}, "
                 f"random-vector sd = {1 / np.sqrt(DIM):.3f}", fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    print(f"figure    -> {out}.png / .pdf")


def cmd_build(args):
    if args.seed != SEED:
        print(f"WARNING: --seed {args.seed} != locked SEED {SEED}; "
              f"the resulting criteria.npz will NOT match the pipeline")
    vocab = make_vocabulary(args.seed, DIM)
    Xi = build_criteria(vocab)
    names = np.array(list(CRITERIA))

    S = similarity_matrix(Xi)
    off = S - np.eye(len(names))
    max_off = float(np.abs(off).max())

    print(f"builder {BUILDER_VERSION}  dim={DIM}  seed={args.seed}")
    print("\n6x6 cosine similarity (report figure):")
    hdr = "".join(f"{n[:10]:>12s}" for n in names)
    print(" " * 24 + hdr)
    for i, n in enumerate(names):
        row = "".join(f"{S[i, j]:12.3f}" for j in range(len(names)))
        print(f"{n[:22]:>24s}{row}")
    print(f"\nmax |off-diagonal| = {max_off:.3f}  "
          f"(random-vector sd at d={DIM}: {1 / np.sqrt(DIM):.3f})")
    if max_off >= MAX_OFFDIAG:
        raise SystemExit(
            f"criteria insufficiently separated ({max_off:.3f} >= "
            f"{MAX_OFFDIAG}); change SEED in criteria_builder.py and rebuild")

    audit = unbinding_audit(Xi, vocab)
    bad = [a["criterion"] for a in audit if not a["separated"]]
    if bad:
        raise SystemExit(f"unbinding audit failed for: {bad}")
    print("unbinding audit: all 6 criteria recover their own terms above "
          "the best foreign term")

    np.savez(args.out, criteria=Xi, names=names)
    manifest = {
        "builder": BUILDER_VERSION,
        "dim": DIM, "seed": args.seed,
        "hrr_implementation": "Kelly & Tomkins-Flanagan, ecphory/hrr "
                              "(Plate 1995)",
        "construction": "xi_k = unit(name_k + sum_j name_k (*) term_j); "
                        "circular convolution binding, superposition",
        "criteria_terms": dict(CRITERIA),
        "row_order": list(CRITERIA),
        "max_offdiagonal_cosine": round(max_off, 4),
        "unbinding_audit": audit,
    }
    mpath = Path(args.out).with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    similarity_figure(S, list(names))
    print(f"\ncriteria  -> {args.out}   ({Xi.shape}, float32, unit rows)")
    print(f"manifest  -> {mpath}   <- the audit-trail table for the appendix")


def cmd_audit(args):
    z = np.load(args.criteria)
    Xi, names = z["criteria"], list(z["names"])
    if names != list(CRITERIA):
        raise SystemExit(
            "criteria.npz row order does not match this builder's CRITERIA dict")
    if args.seed != SEED:
        print(f"WARNING: auditing with --seed {args.seed} != locked SEED "
              f"{SEED}; drift below will be meaningless")
    vocab = make_vocabulary(args.seed, DIM)
    rebuilt = build_criteria(vocab)
    drift = float(np.abs(rebuilt - Xi).max())
    print(f"max |rebuilt - stored| = {drift:.2e}  "
          f"(0 means the file matches this code+seed exactly)")
    if drift > 1e-4:
        raise SystemExit(
            f"criteria.npz does not match this builder (drift {drift:.2e}); "
            f"it was built with different code, seed, or DIM -- downstream "
            f"runs against it are not trustworthy") 
    for a in unbinding_audit(Xi, vocab):
        own = ", ".join(f"{t}={s}" for t, s in a["own_terms"].items())
        print(f"\n{a['criterion']}")
        print(f"  own:     {own}")
        print(f"  foreign: best {a['top_foreign_term']} = "
              f"{a['top_foreign_sim']}   separated: {a['separated']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--out", default="criteria.npz")
    b.add_argument("--seed", type=int, default=SEED)
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("audit")
    a.add_argument("--criteria", default="criteria.npz")
    a.add_argument("--seed", type=int, default=SEED)
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
