"""
Environment and GPU report check after installing all requirements.
"""

import argparse
import json
import importlib
import platform
import sys
from pathlib import Path

OK = "OK  "
WARN = "WARN"
FAIL = "FAIL"

# transformers 4.48.0 added the ModernBERT architecture
MIN_TRANSFORMERS = (4, 48)
# bfloat16 has hardware support from NVIDIA Ampere onward to reduce memory requirements
BF16_MIN_CAPABILITY = (8, 0)


def _ver_tuple(v: str):
    """Leading numbers of a version string: '4.48.1' -> (4, 48, 1)."""
    out = []
    for part in v.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def line(status, label, value, note=""):
    print(f"  [{status}] {label:22s} {value}" + (f"   <- {note}" if note else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="logs/env.json",
                    help="where to write the provenance record")
    args = ap.parse_args()

    report = {"python": sys.version.split()[0],
              "platform": platform.platform()}
    problems, warnings = [], []

    print("\nenvironment")
    line(OK, "python", report["python"])
    line(OK, "platform", report["platform"])

    # ---------------------------------------------------------------- torch
    print("\ntorch")
    try:
        import torch
    except ImportError:
        line(FAIL, "torch", "not installed")
        sys.exit(1)

    report["torch"] = torch.__version__
    report["torch_cuda_build"] = torch.version.cuda
    line(OK, "torch", torch.__version__)

    if torch.version.cuda is None:
        # The version string confirms: "+cpu" vs "+cu130".
        line(FAIL, "cuda build", "CPU-ONLY WHEEL",
             "this will run, slowly, and never say why")
        if platform.system() == "Windows":
            note = (" On Windows `pip install torch` gives the "
                    "CPU build -- the index URL is not optional here.")
        problems.append(
            f"torch {torch.__version__} is the CPU-only wheel (the '+cpu' "
            f"suffix): no CUDA support compiled in. Reinstall from the "
            f"CUDA index:\n    pip install --force-reinstall torch "
            f"--index-url https://download.pytorch.org/whl/cu130")
    else:
        line(OK, "cuda build", f"compiled against CUDA {torch.version.cuda}",
             "the '+cuXXX' suffix above is the confirmation")

    # ------------------------------------------------------------ the device
    print("\ngpu")
    available = torch.cuda.is_available()
    report["cuda_available"] = bool(available)

    if not available:
        if torch.version.cuda is None:
            line(FAIL, "cuda.is_available", "False", "CPU-only wheel, above")
        else:
            line(FAIL, "cuda.is_available", "False",
                 "CUDA build present but no device visible")
            problems.append(
                "torch is a CUDA build but cannot see a device.")
        report["device"] = "cpu"
    else:
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        report.update({"device": "cuda", "device_count": n,
                       "device_name": name,
                       "capability": f"{cap[0]}.{cap[1]}",
                       "vram_gb": round(total_gb, 1)})

        line(OK, "cuda.is_available", "True")
        line(OK, "device", f"{name}  ({n} visible)")
        line(OK, "compute capability", f"sm_{cap[0]}{cap[1]}")
        line(OK, "vram", f"{total_gb:.1f} GB")

        # --- bfloat16 used by embed.py and rq3_pipeline ---------
        bf16_native = cap >= BF16_MIN_CAPABILITY
        report["bf16_native"] = bool(bf16_native)
        if bf16_native:
            line(OK, "bfloat16", "hardware support (Ampere or newer)",
                 "autocast paths will be fast")
        else:
            line(WARN, "bfloat16", f"emulated on sm_{cap[0]}{cap[1]}",
                 "autocast will not give a speedup")
            warnings.append(
                f"This GPU (sm_{cap[0]}{cap[1]}) predates hardware bfloat16.")

        # --- test ----------------------------------------------
        try:
            x = torch.randn(64, 64, device="cuda") # allocates in GPU memory
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            line(OK, "allocation test", "passed", "a real matmul ran on device")
            report["gpu_allocation_test"] = True
        except Exception as exc:                       # noqa: BLE001
            line(FAIL, "allocation test", type(exc).__name__, str(exc)[:60])
            report["gpu_allocation_test"] = False
            problems.append(
                f"torch sees a GPU but could not run on it: {exc}.")

    # --------------------------------------------------------- the libraries
    print("\nlibraries")
    for mod, floor, hard in (("transformers", MIN_TRANSFORMERS, True),
                             ("datasets", None, True),
                             ("numpy", None, True),
                             ("matplotlib", None, False)):
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "unknown")
            report[mod] = v
            if floor and _ver_tuple(v) < floor:
                want = ".".join(str(p) for p in floor)
                line(FAIL, mod, v, f"need >= {want} for ModernBERT")
                problems.append(
                    f"{mod} {v} is too old; ModernBERT needs >= {want}.")
            else:
                line(OK, mod, v)
        except ImportError:
            report[mod] = None
            if hard:
                line(FAIL, mod, "not installed")
                problems.append(f"{mod} is required. pip install -r requirements.txt")
            else:
                line(WARN, mod, "not installed",
                     "figures will be skipped, pipeline still runs")
                warnings.append(
                    f"{mod} is missing. Every figure module prints and skips.")

    # ------------------------------------------------------------- ollama
    
    print("\nollama (run 3 only)")
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags",
                                    timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        models = [m.get("name") or m.get("model") for m in tags.get("models", [])]
        report["ollama_models"] = models
        line(OK, "ollama", f"running, {len(models)} model(s)",
             ", ".join(models[:3]) if models else "none pulled yet")
        if not models:
            warnings.append(
                "Ollama is running but has no models pulled.")
    except Exception:                                  # noqa: BLE001
        report["ollama_models"] = None
        line(WARN, "ollama", "not reachable at localhost:11434",
             "only needed for run 3")

    # -------------------------------------------------------------- results
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nprovenance -> {out}")
    print("  (quote torch, transformers, device_name and capability in the "
          "implementation paragraph)")

    if warnings:
        print("\n" + "-" * 70)
        for w in warnings:
            print(f"WARNING: {w}\n")

    if problems:
        print("-" * 70)
        for p in problems:
            print(f"PROBLEM: {p}\n")
        print(f"{len(problems)} problem(s). Fix before running run_1_setup.py.")
        sys.exit(1)

    print("\nEnvironment is ready.")


if __name__ == "__main__":
    main()
