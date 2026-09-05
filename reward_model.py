"""
Energy-based reward model, single source of trust for the architecture.

    python reward_model.py test
    python reward_model.py train --model hrr    --emb-train emb_train.npz --emb-test emb_test.npz --criteria criteria.npz
    python reward_model.py train --model scalar --emb-train emb_train.npz --emb-test emb_test.npz
    python reward_model.py eval  --checkpoint runs/hrr/checkpoint.pt --emb-test emb_test.npz

ARCHITECTURE (all decisions locked; do not drift)
-------------------------------------------------
encoder     ModernBERT-base is frozen and revision-pinned. Each input is processed
            exactly once. The CLS-token embeddings output generated from embed.py
            are cached to a .npz file. Any required retraining is fast, leveraging
            the static cache instead of re-encoding the text.
            The design is for a frozen ModernBERT encoder that contributes no
            trainable parameters to the reward model. Any performance differences
            between the HRR and SCALAR heads are thus attributable to the reward
            head rather than the representation.

  W_Q       the only trainable component of the HRR model is a 768x768 linear
            map (no bias) translating encoder space into criteria space.

  Xi        6x768 stored-pattern matrix from the HRR criteria builder,
            frozen, registered as a buffer so no optimizer can ever see it.
            The rows are named, auditable criteria, so freezing them is the
            the interpretable-by-construction claim, since a learned Xi
            would un-name them.

  W_K, W_V  are deliberately absent. Ramsauer's Equation 10 introduces them
            to draw the correspondence with transformer attention; here the
            keys ARE the stored patterns, and p is read out directly rather
            than projected through a value matrix.

  reward = -E, with E from Ramsauer et al. (2020), Equation 2:

    E = -(1/beta) * lse(beta * Xi q)
        + 0.5 * ||q||^2
        + (1/beta) * log K
        + 0.5 * M^2
          
    M = max_k ||xi_k||

  All four terms are computed, including the two constants (M = 1 here,
  since the criterion rows are unit-normalised), so the reported reward is
  literally -E rather than "-E excluding constants". The constants cancel in
  the Bradley-Terry loss, which sees only reward differences, so training
  is unaffected either way.

  Ramsauer et al. (2020), Equation 3
  attribution p = softmax(beta * Xi q)

  p and the reward are computed in the same forward pass, from the same
  Xi-dot-q product. That simultaneity is the interpretable-by-construction
  claim. The attribution is not a post-hoc explanation of the score, it is
  part of the arithmetic that produced it.

BASELINE
--------
  ScalarReward: Linear(768 -> 1) on the same frozen embeddings, same loss,
  same training budget. The comparison in RQ1/RQ3 is head vs head with
  everything else matched.

LOSS
----
  Bradley-Terry: -log sigmoid(r_chosen - r_rejected). The sole training
  signal (Christiano et al. 2017; Ouyang et al. 2022 for its use in RLHF).

MODEL SELECTION
---------------
  A validation slice is created from the END of train_prefs (--val-frac).
  Selection happens on validation accuracy, and test_prefs is scored by
  `eval`, once, at the end.

DETERMINISM AND TORCH VERSION
-----------------------------
  Every weight initialisation here draws from a LOCAL torch.Generator seeded
  from --seed, so a run is reproducible without depending on global RNG
  state, and constructing both reward heads in one process cannot correlate
  them. cmd_train additionally calls torch.manual_seed for anything outside
  these constructors.

  Initialisation uses a local Glorot helper (_xavier_uniform_) rather than
  nn.init.xavier_uniform_, so seeded init is identical regardless of the
  torch version the code is run on.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_VERSION = "reward_model/v1"
EMB_DIM = 768
K_CRITERIA = 6

# beta is the inverse temperature of the Hopfield read-out. Ramsauer's
# transformer paper uses 1/sqrt(d); with d=768 that is ~0.036. It is
# a reported hyperparameter, sets retrieval sharpness and thus shapes
# the k-count statistic in RQ2.

DEFAULT_BETA = 1.0 / math.sqrt(EMB_DIM)


# ------------------------------------------------------------------- init

def _xavier_uniform_(w: torch.Tensor, gen: torch.Generator,
                     gain: float = 1.0) -> torch.Tensor:
    """
    Glorot uniform initialisation, seeded from a local generator.

    Identical to nn.init.xavier_uniform_: bound a = gain*sqrt(6/(fan_in +
    fan_out)) with fan_in = w.shape[-1] and fan_out = w.shape[-2]
    """
    fan_out, fan_in = w.shape[-2], w.shape[-1]
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        return w.uniform_(-a, a, generator=gen)


# ------------------------------------------------------------------ models

class HopfieldEnergyReward(nn.Module):
    """
    Frozen stored patterns Xi (K x d), learned query projection W_Q (d x d),
    reward = -E per Ramsauer Equation 2
    attribution p per Ramsauer Equation 3
    """

    def __init__(self, criteria: np.ndarray, beta: float = DEFAULT_BETA,
                 init: str = "identity", seed: int = 42):
        super().__init__()
        if criteria.shape != (K_CRITERIA, EMB_DIM):
            raise ValueError(f"criteria must be {K_CRITERIA}x{EMB_DIM}, "
                             f"got {criteria.shape}")

        # register_buffer => saved with the model, moved with .to(device),
        # and opaque to optimizers. Xi cannot drift by accident.
        self.register_buffer("Xi", torch.tensor(criteria, dtype=torch.float32))
        self.beta = float(beta)

        # create the W_Q translation matrix
        self.W_Q = nn.Linear(EMB_DIM, EMB_DIM, bias=False)
        g = torch.Generator().manual_seed(seed)
        if init == "identity":
            # q starts as the raw embedding, the model begins by scoring
            # untranslated encoder space against the criteria, and training
            # learns the translation.
            # Initial small noise breaks symmetry.
            with torch.no_grad():
                self.W_Q.weight.copy_(
                    torch.eye(EMB_DIM) # identity matrix
                    + 0.01 * torch.randn(EMB_DIM, EMB_DIM, generator=g)) # plus random noise
        elif init == "random":
            with torch.no_grad():
                w = torch.empty(EMB_DIM, EMB_DIM)
                _xavier_uniform_(w, g)
                self.W_Q.weight.copy_(w)
        else:
            raise ValueError(f"unknown init: {init}")

        # constants of Equation 2, fixed once because Xi is frozen
        self._log_K = math.log(K_CRITERIA)
        self._half_M2 = 0.5 * float((self.Xi.norm(dim=1).max()) ** 2)

    def forward(self, emb: torch.Tensor):
        """
        emb: (B, d) pooled encoder embeddings.
        returns reward (B,), attribution p (B, K), logits beta*Xi q (B, K)
        """
        q = self.W_Q(emb) # (B, d); embeddings translated into criteria space
        logits = self.beta * (q @ self.Xi.T) # (B, K); beta-scaled dot products of query with each stored pattern
        # Ramsauer Equation 2, four terms: log-sum-exp of the logits, scaled back by beta
        lse = torch.logsumexp(logits, dim=-1) / self.beta
        half_q2 = 0.5 * (q * q).sum(dim=-1)
        energy = -lse + half_q2 + self._log_K / self.beta + self._half_M2
        reward = -energy
        p = F.softmax(logits, dim=-1) # Ramsauer Equation 3, same pass
        return reward, p, logits


class ScalarReward(nn.Module):
    """Matched baseline: one linear head, same embeddings, same loss."""

    def __init__(self, seed: int = 42):
        super().__init__()
        self.head = nn.Linear(EMB_DIM, 1, bias=True)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            w = torch.empty(1, EMB_DIM)
            _xavier_uniform_(w, g)
            self.head.weight.copy_(w)
            self.head.bias.zero_()

    def forward(self, emb: torch.Tensor):
        reward = self.head(emb).squeeze(-1)
        return reward, None, None

# Bradley-Terry Loss, called line 298
def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor):
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def k_count(p: torch.Tensor, mass: float = 0.90) -> torch.Tensor:
    """
    RQ2 statistic is the number of criteria needed to cover
    `mass` of the attribution. 1 = retrieval of a single
    criterion; 6 = uniform blur. Reported per input.
    """
    sorted_p, _ = torch.sort(p, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    return (cum < mass).sum(dim=-1) + 1


# ------------------------------------------------------------------- data

def load_pairs(path: str):
    """
    embed.py writes npz with emb_chosen, emb_rejected (N, 768) float32
    plus a JSON sidecar of provenance.
    """
    z = np.load(path)
    c, r = z["emb_chosen"], z["emb_rejected"]
    if c.shape != r.shape or c.shape[1] != EMB_DIM:
        raise ValueError(f"bad embedding shapes: {c.shape} / {r.shape}")
    meta = {}
    side = Path(path).with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
    return torch.tensor(c), torch.tensor(r), meta


@torch.no_grad()
def pair_accuracy(model, emb_c, emb_r, device, batch=4096):
    model.eval()
    correct, n = 0, emb_c.shape[0]
    for i in range(0, n, batch):
        rc, _, _ = model(emb_c[i:i + batch].to(device))
        rr, _, _ = model(emb_r[i:i + batch].to(device))
        correct += (rc > rr).sum().item()
    return correct / n


# ---------------------------------------------------------------- training

def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    emb_c, emb_r, meta = load_pairs(args.emb_train)
    n = emb_c.shape[0]

    # validation slice from the END of train; test stays untouched here
    n_val = max(1, int(n * args.val_frac))
    val_c, val_r = emb_c[n - n_val:], emb_r[n - n_val:]
    emb_c, emb_r = emb_c[:n - n_val], emb_r[:n - n_val]
    n = emb_c.shape[0]

    if args.model == "hrr":
        crit = np.load(args.criteria)
        Xi = crit["criteria"] if "criteria" in crit else crit[crit.files[0]]
        crit_names = list(crit["names"]) if "names" in crit else None
        model = HopfieldEnergyReward(Xi, beta=args.beta,
                                     init=args.init, seed=args.seed)
    else:
        crit_names = None
        model = ScalarReward(seed=args.seed)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            weight_decay=args.weight_decay)

    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  trainable params={n_params:,}  device={device}")
    print(f"torch={torch.__version__}  seed={args.seed}")
    print(f"train pairs={n:,}  val pairs={n_val:,}")

    # 20 epochs and 53 total batches
    best_val, best_state, best_epoch, t0 = -1.0, None, -1, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(
            args.seed + epoch))
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch] # shuffle and batch
            rc, _, _ = model(emb_c[idx].to(device)) # chosen scored
            rr, _, _ = model(emb_r[idx].to(device)) # rejected scored
            loss = bradley_terry_loss(rc, rr)
            opt.zero_grad() # one update per batch
            loss.backward() # one update per batch
            opt.step() # one update per batch
            total += loss.item() * len(idx)

        # validation passes through the model, but no gradient, no optimizer step
        val_acc = pair_accuracy(model, val_c, val_r, device)
        marker = ""
        # best epoch is checkpointed, so the final model is the best one seen
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  <- best"
        # total/n is the per-example mean (sum of loss*batch_size, over n),
        # not a plain average of per-batch means -- the final batch is a
        # different size, so the two would differ slightly. Not a bug.
        print(f"epoch {epoch:3d}  loss {total / n:.4f}  "
              f"val acc {val_acc:.4f}{marker}")

    # checkpoint the best model, plus metadata for reproducibility and auditing
    model.load_state_dict(best_state)
    ckpt_path = outdir / "checkpoint.pt"
    beta_val = getattr(model, "beta", None)
    torch.save({"model_version": MODEL_VERSION,
                "model_type": args.model,
                "state_dict": best_state,
                "beta": beta_val,
                "criteria_names": crit_names,
                "seed": args.seed,
                "best_epoch": best_epoch}, ckpt_path)

    manifest = {
        "model_version": MODEL_VERSION,
        "model_type": args.model,
        "trainable_params": n_params,
        "beta": beta_val,
        "init": args.init if args.model == "hrr" else None,
        "criteria_file": args.criteria if args.model == "hrr" else None,
        "criteria_names": crit_names,
        "seed": args.seed,
        "torch": torch.__version__,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "weight_decay": args.weight_decay, "val_frac": args.val_frac,
        "best_val_accuracy": best_val,
        "best_epoch": best_epoch,
        "train_pairs": n, "val_pairs": n_val,
        "embedding_meta": meta,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nbest val accuracy {best_val:.4f}  (epoch {best_epoch}/{args.epochs})")
    print(f"checkpoint -> {ckpt_path}")
    print(f"manifest   -> {outdir / 'manifest.json'}")
    print("test_prefs is scored by `eval`, once. Do not select on it.")


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["model_type"] == "hrr":
        Xi = ck["state_dict"]["Xi"].numpy()
        model = HopfieldEnergyReward(Xi, beta=ck["beta"], seed=ck["seed"])
    else:
        model = ScalarReward(seed=ck["seed"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def cmd_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.checkpoint, device)
    emb_c, emb_r, meta = load_pairs(args.emb_test)
    acc = pair_accuracy(model, emb_c, emb_r, device)
    print(f"model={ck['model_type']}  test pairs={emb_c.shape[0]:,}")
    print(f"TEST pairwise accuracy: {acc:.4f}")
    if ck["model_type"] == "hrr":
        with torch.no_grad():
            _, p, _ = model(emb_c[:4096].to(device))
            kc = k_count(p).float()
        print(f"k-count on chosen (first 4096): "
              f"mean {kc.mean():.2f}  median {kc.median():.0f}")


# --------------------------------------------------------------- self-test

def cmd_test(_args):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # unit-norm rows, like the HRR builder produces
    Xi = rng.normal(size=(K_CRITERIA, EMB_DIM)).astype(np.float32)
    Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)

    model = HopfieldEnergyReward(Xi)
    emb = torch.randn(32, EMB_DIM)
    r, p, logits = model(emb)

    assert r.shape == (32,), r.shape
    assert p.shape == (32, K_CRITERIA), p.shape
    assert torch.allclose(p.sum(-1), torch.ones(32), atol=1e-5), "p must sum to 1"

    kc = k_count(p)
    assert ((kc >= 1) & (kc <= K_CRITERIA)).all(), "k-count out of range"

    # Xi is frozen: not a parameter, receives no gradient
    assert all(name != "Xi" for name, _ in model.named_parameters())
    loss = bradley_terry_loss(r[:16], r[16:])
    loss.backward()
    assert model.W_Q.weight.grad is not None, "W_Q must receive gradient"

    # the ONLY trainable tensor is W_Q: exactly d*d parameters
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    assert n_train == EMB_DIM * EMB_DIM, n_train

    # training reduces BT loss on a learnable toy problem
    emb_c = torch.randn(256, EMB_DIM) + 0.3 * torch.tensor(Xi[0])
    emb_r = torch.randn(256, EMB_DIM)
    m2 = HopfieldEnergyReward(Xi)
    opt = torch.optim.AdamW([p_ for p_ in m2.parameters() if p_.requires_grad],
                            lr=1e-3)
    losses = []
    for _ in range(60):
        rc, _, _ = m2(emb_c)
        rr, _, _ = m2(emb_r)
        L = bradley_terry_loss(rc, rr)
        opt.zero_grad(); L.backward(); opt.step()
        losses.append(L.item())
    assert losses[-1] < losses[0], (losses[0], losses[-1])

    # scalar baseline: matched interface, 769 params (w + b)
    sm = ScalarReward()
    rs, ps, _ = sm(emb)
    assert rs.shape == (32,) and ps is None
    assert sum(p_.numel() for p_ in sm.parameters()) == EMB_DIM + 1

    # identity init sanity: at t=0, q ~ emb, so rewards from raw space
    mi = HopfieldEnergyReward(Xi, init="identity", seed=1)
    q0 = mi.W_Q(emb)
    assert (q0 - emb).abs().mean() < 0.5, "identity init should start near q=e"

    # --- initialisation is version-independent and locally seeded ---------

    s1, s2 = ScalarReward(seed=7), ScalarReward(seed=7)
    assert torch.equal(s1.head.weight, s2.head.weight), \
        "scalar init must be reproducible from --seed alone"
    assert not torch.equal(ScalarReward(seed=8).head.weight, s1.head.weight), \
        "different seeds must give different scalar inits"

    # Glorot bound: every sampled weight inside +/- sqrt(6/(fan_in+fan_out))
    bound = math.sqrt(6.0 / (EMB_DIM + 1))
    assert s1.head.weight.abs().max() <= bound, "xavier bound violated"

    # HRR random-initialization branch constructs
    mr1 = HopfieldEnergyReward(Xi, init="random", seed=7)
    mr2 = HopfieldEnergyReward(Xi, init="random", seed=7)
    assert torch.equal(mr1.W_Q.weight, mr2.W_Q.weight), \
        "hrr random init must be reproducible from --seed alone"

    # Local generators: the weights a model ends up with must depend on the --seed

    torch.manual_seed(123)
    s_a = ScalarReward(seed=99).head.weight.clone()
    h_a = HopfieldEnergyReward(Xi, seed=99).W_Q.weight.clone()
    torch.manual_seed(456)
    s_b = ScalarReward(seed=99).head.weight.clone()
    h_b = HopfieldEnergyReward(Xi, seed=99).W_Q.weight.clone()
    assert torch.equal(s_a, s_b), \
        "scalar init must not depend on global RNG state"
    assert torch.equal(h_a, h_b), \
        "hrr identity init must not depend on global RNG state"
    r_a = HopfieldEnergyReward(Xi, init="random", seed=99).W_Q.weight.clone()
    torch.manual_seed(789)
    r_b = HopfieldEnergyReward(Xi, init="random", seed=99).W_Q.weight.clone()
    assert torch.equal(r_a, r_b), \
        "hrr random init must not depend on global RNG state"

    print("all self-tests passed")
    print(f"  trainable (hrr): {n_train:,} = 768x768 (W_Q only)")
    print(f"  trainable (scalar): {EMB_DIM + 1:,}")
    print(f"  toy BT loss: {losses[0]:.4f} -> {losses[-1]:.4f} over 60 steps")
    print(f"  init: locally seeded, torch {torch.__version__}, "
          f"no nn.init generator dependency")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train is the main training loop, saving a checkpoint and manifest
    t = sub.add_parser("train")
    t.add_argument("--model", choices=["hrr", "scalar"], required=True)
    t.add_argument("--emb-train", required=True)
    t.add_argument("--emb-test", required=False,
                   help="unused during training; kept to catch path typos early")
    t.add_argument("--criteria", default="criteria.npz",
                   help="npz with 'criteria' (6x768) and 'names' (6,)")
    t.add_argument("--beta", type=float, default=DEFAULT_BETA)
    t.add_argument("--init", choices=["identity", "random"], default="identity")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--outdir", default="runs")
    t.set_defaults(func=cmd_train)

    # eval is a single-shot evaluation of a checkpoint on test_prefs
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--emb-test", required=True)
    e.set_defaults(func=cmd_eval)

    # test is a self-contained unit test, no data needed, no checkpoint needed
    s = sub.add_parser("test", help="self-tests, no data needed")
    s.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
