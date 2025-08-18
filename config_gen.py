import argparse
import os
import re
from typing import Dict, List

import pandas as pd
import yaml


def detect_model_size_hint(path: str) -> str:
    """Detect model size hint from a path string.

    Returns one of: '7', '8', '13', '30'. Defaults to '13' if unknown.
    """
    s = path.lower()
    if re.search(r"(?<!\d)30(?!\d)|30b|llama-1-30", s):
        return "30"
    if re.search(r"(?<!\d)13(?!\d)|13b", s):
        return "13"
    if re.search(r"(?<!\d)8(?!\d)|8b", s):
        return "8"
    if re.search(r"(?<!\d)7(?!\d)|7b", s):
        return "7"
    return "13"


def get_param_proportion(model_size: str) -> Dict[str, float]:
    """Return parameter proportion per layer-type for the given model size.

    Values mirror params.get_param_propotion but kept self-contained here and
    include aggregated keys 'qkv_proj' and 'gateup_proj' when applicable.
    Returned proportions are fractions of the whole model (sum across all  layers ~ 1.0).
    """
    print(f"Model size: {model_size}")
    if model_size == "7":
        return {
            "down_proj": 0.222797927 / 32,
            "up_proj": 0.222797927 / 32,
            "gate_proj": 0.222797927 / 32,
            "q_proj": 0.082901554 / 32,
            "k_proj": 0.082901554 / 32,
            "v_proj": 0.082901554 / 32,
            "o_proj": 0.082901554 / 32,
            "qkv_proj": (0.082901554 / 32) * 3,
            "gateup_proj": (0.222797927 / 32) * 2,
        }
    if model_size == "13":
        return {
            "down_proj": 0.2230543849 / 40,
            "up_proj": 0.2230543849 / 40,
            "gate_proj": 0.2230543849 / 40,
            "q_proj": 0.08270921129 / 40,
            "k_proj": 0.08270921129 / 40,
            "v_proj": 0.08270921129 / 40,
            "o_proj": 0.08270921129 / 40,
            "qkv_proj": 0.08270921129/40 * 3,
            "gateup_proj": 0.2230543849/40 * 2,
        }

    return ValueError(f"Unknown model size: {model_size}. Supported sizes are: 7, 8, 13, 30.")

def rename_key(old_key: str) -> str | None:
    """Convert 'layer_12_qkv_proj' to 'model.layers.12.self_attn.qkv_proj'.
    Returns None if malformed.
    """
    parts = old_key.split("_")
    if len(parts) < 3 or parts[0] != "layer":
        return None
    layer_num = parts[1]
    proj_type = "_".join(parts[2:])

    # Determine submodule
    if proj_type in {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}:
        submodule = "self_attn"
    elif proj_type in {"gate_proj", "up_proj", "down_proj", "gateup_proj"}:
        submodule = "mlp"
    else:
        # Unknown; skip
        return None

    return f"model.layers.{layer_num}.{submodule}.{proj_type}"


def layer_cost(layer_name: str, type_costs: Dict[str, float]) -> float:
    """Get the cost of a full layer name using the provided type proportions."""
    # Identify the final proj type
    proj_type = layer_name.split(".")[-1]
    return type_costs.get(proj_type, 0.0)


def pick_sensitive_layers(
    scores: Dict[str, float],
    budget: float,
    type_costs: Dict[str, float],
    higher_is_worse: bool = True,
) -> List[str]:
    """Greedy pick layers by score until budget exhausted.

    scores: mapping from full layer name to oracle score.
    budget: total parameter fraction available for sensitive layers.
    type_costs: mapping from proj type to parameter fraction cost.
    higher_is_worse: if True, sort descending (e.g., KLD), else ascending.
    """
    # Sort by score
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=higher_is_worse)

    selected: List[str] = []
    total = 0.0
    for lname, _ in items:
        c = layer_cost(lname, type_costs)
        if c <= 0:
            continue
        if total + c <= budget:
            selected.append(lname)
            total += c
        if total >= budget:
            break
    print(f"Total: {total}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate sensitive_layers YAML by KLD under budget")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: Layer, KLD")
    parser.add_argument("--output", default="sensitive_layers.yaml", help="Output YAML path")
    parser.add_argument("--precision", default="w8a8", help="Precision string to store in YAML")
    # Budget options
    parser.add_argument("--budget", type=float, default=None, help="Budget fraction (0..1) for sensitive layers")
    parser.add_argument(
        "--up-rate",
        type=float,
        default=None,
        help="Alternate name for budget fraction (0..1) for sensitive layers",
    )
    parser.add_argument(
        "--quant-rate",
        type=float,
        default=None,
        help="If provided, budget will be computed as 1 - quant_rate",
    )
    parser.add_argument(
        "--model-size",
        choices=["7", "8", "13", "30"],
        default=None,
        help="Override model size detection (7/8/13/30)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    # Determine budget
    budget_vals = [v for v in [args.budget, args.up_rate, (1 - args.quant_rate) if args.quant_rate else None] if v is not None]
    if not budget_vals:
        raise ValueError("Provide one of --budget, --up-rate, or --quant-rate")
    budget = budget_vals[0]
    if not (0.0 <= budget <= 1.0):
        raise ValueError(f"Budget must be in [0,1], got {budget}")

    # Detect model size
    model_size = args.model_size or detect_model_size_hint(args.csv)
    type_costs = get_param_proportion(model_size)

    # Load CSV and build scores
    df = pd.read_csv(args.csv)
    if "Layer" not in df.columns or "KLD" not in df.columns:
        raise ValueError("CSV must contain 'Layer' and 'KLD' columns")
    # Filter out NaNs
    df = df[df["KLD"].notna() & df["Layer"].notna()]

    # Map names
    scores: Dict[str, float] = {}
    for old_key, kld in zip(df["Layer"].tolist(), df["KLD"].tolist()):
        new_key = rename_key(str(old_key))
        if new_key is None:
            continue
        scores[new_key] = float(kld)

    # Greedy selection (higher KLD means worse -> more sensitive)
    sensitive_layers = pick_sensitive_layers(scores, budget=budget, type_costs=type_costs, higher_is_worse=True)

    data = {
        "precision": args.precision,
        "sensitive_layers": sensitive_layers,
    }

    with open(args.output, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Wrote {len(sensitive_layers)} sensitive layers to {args.output} (budget={budget:.4f}, model_size={model_size})")


if __name__ == "__main__":
    main()
