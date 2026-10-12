"""Print real trees from a trained model, to see what questions it actually asks.

    python -m src.show_tree                                        # first 3 trees, volunteer 1h model
    python -m src.show_tree --tag volunteer_h1_from20251101 --trees 0 1 50 --depth 3

Each node shows the question (feature <= threshold) and how many training rows
reached it; each leaf shows the score ("bal") this tree adds. Scores are in
log-odds: the model sums the baseline and the scores of all trees, then turns the
sum into a probability with p = 1 / (1 + exp(-sum)).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def internal_feature_names(model, features):
    """Feature names in the model's INTERNAL column order.

    With categorical features, sklearn's HistGradientBoosting reorders columns
    (categorical first, then numerical), and node["feature_idx"] refers to that
    internal order, not to the order of FEATURES.
    """
    pre = getattr(model, "_preprocessor", None)
    if pre is None:
        return list(features)
    names = []
    for name, _, mask in pre.transformers_:
        if name == "remainder":
            continue
        names += [f for f, keep in zip(features, mask) if keep]
    return names


def show(nodes, features, i=0, depth=0, max_depth=3, prefix=""):
    n = nodes[i]
    pad = "    " * depth
    if n["is_leaf"]:
        print(f"{pad}{prefix}leaf: score {n['value']:+.3f}   (rows: {n['count']})")
        return
    name = features[n["feature_idx"]]
    if depth >= max_depth:
        print(f"{pad}{prefix}... ({name} ...; rows: {n['count']})")
        return
    if n["is_categorical"]:
        q = f"{name} in some set of regions?"
    else:
        q = f"{name} <= {n['num_threshold']:.3g} ?"
    print(f"{pad}{prefix}{q}   (rows: {n['count']})")
    show(nodes, features, n["left"], depth + 1, max_depth, "yes -> ")
    show(nodes, features, n["right"], depth + 1, max_depth, "no  -> ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="volunteer_h1_from20220415")
    ap.add_argument("--trees", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / f"model_{args.tag}.joblib")
    model = bundle["model"]
    features = internal_feature_names(model, bundle["features"])
    base = float(np.ravel(model._baseline_prediction)[0])
    print(f"model {args.tag}: {len(model._predictors)} trees")
    print(f"baseline score {base:+.3f} = probability {1 / (1 + np.exp(-base)):.3f} before any tree\n")
    for k in args.trees:
        print(f"--- tree {k} ---")
        show(model._predictors[k][0].nodes, features, max_depth=args.depth)
        print()


if __name__ == "__main__":
    main()
