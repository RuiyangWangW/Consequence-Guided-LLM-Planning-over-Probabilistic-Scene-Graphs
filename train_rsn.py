"""Train a SEEK-style Relational Semantic Network on BEHAVIOR-1K.

Follows the RSN architecture from Ginting et al., "SEEK: Semantic Reasoning for Object
Goal Navigation in Real World Inspection Tasks" (RSS 2024), section IV-B:

    object name -> frozen text encoder -> MLP (3 hidden layers) -> P(object | room)
                                                                   for every room type

Two properties of this design carry the weight:

1. **A frozen text embedding, not a learned per-category embedding.** SEEK's
   open-vocabulary behavior comes entirely from the text encoder: an object absent from
   training still lands near similar objects in embedding space, so the model
   generalizes instead of failing. A learned embedding can only answer for the exact
   categories it was trained on, which is unworkable when an LLM can name any object.

2. **The room is an output dimension, not an input.** SEEK emits a vector over all room
   types in one forward pass, which is what its MDP planner consumes (it needs
   P(object) in *every* room to compute a policy). One pass per object instead of one
   per (object, room) pair.

Where we deliberately diverge from SEEK, and why:

- **Loss.** SEEK regresses with MSE onto soft probabilities distilled from GPT-4. We
  have hard occurrence labels from real scene layouts, so we use BCE, which is the
  correct likelihood for binary observations. Reported MSE (Brier score) alongside it
  keeps the numbers comparable to the paper's Table II.
- **Second output head.** SEEK also predicts P(find without careful search), used to
  set MDP transition probabilities. That quantity is about search difficulty, not
  placement plausibility, and BEHAVIOR has no label for it, so it is omitted.
- **Supervision source.** SEEK distills from GPT-4 because it has no ground-truth
  layouts. We have 51 annotated scenes, which is stronger evidence than an LLM prior.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from evaluation import average_precision, roc_auc, scene_split


class RelationalSemanticNetwork(nn.Module):
    """Frozen-text-embedding MLP emitting P(object present) for every room type."""

    def __init__(self, embed_dim, n_rooms, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, in_dim = [], embed_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_rooms))
        self.net = nn.Sequential(*layers)

    def forward(self, text_embeddings):
        """(batch, embed_dim) -> (batch, n_rooms) logits."""
        return self.net(text_embeddings)


def build_matrix(rows, categories, room_types):
    """Dense (n_categories, n_rooms) label and mask matrices.

    label[c, r] = 1 if category c occurs in any room instance of type r.
    mask[c, r]  = 1 if that (c, r) pair was observed in this split at all, so rooms
                  absent from a split contribute no gradient.
    """
    cat_idx = {c: i for i, c in enumerate(categories)}
    room_idx = {r: i for i, r in enumerate(room_types)}

    pos = np.zeros((len(categories), len(room_types)), dtype=np.float32)
    seen = np.zeros((len(categories), len(room_types)), dtype=np.float32)
    for r in rows:
        ci, ri = cat_idx[r["object_category"]], room_idx[r["room_type"]]
        seen[ci, ri] = 1.0
        if int(r["label"]):
            pos[ci, ri] = 1.0
    return pos, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default="data/pairs_merged.csv")
    parser.add_argument("--embeddings", default="data/category_embeddings.npz")
    parser.add_argument("--out", default="models/rsn.pt")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="weight on the positive class; default is the full negative/positive ratio, "
        "which maximizes ranking metrics but inflates probabilities. Use 1.0 for "
        "calibrated probabilities (recommended when thresholding).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="fit a temperature/bias on the held-out split after training so predicted "
        "probabilities match observed frequencies (Platt scaling)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    with open(args.pairs) as f:
        rows = list(csv.DictReader(f))

    # Scene-grouped split: validation measures generalization to an unseen building.
    train_rows, val_rows, val_scenes = scene_split(rows, args.val_frac, args.seed)

    cached = np.load(args.embeddings, allow_pickle=True)
    categories = [str(n) for n in cached["names"]]
    emb = torch.from_numpy(cached["embeddings"]).to(device)
    room_types = sorted({r["room_type"] for r in rows})

    data_cats = {r["object_category"] for r in rows}
    missing = data_cats - set(categories)
    if missing:
        raise SystemExit(f"{len(missing)} categories lack embeddings, e.g. {sorted(missing)[:5]}")

    tr_pos, tr_seen = build_matrix(train_rows, categories, room_types)
    va_pos, va_seen = build_matrix(val_rows, categories, room_types)

    tr_y = torch.from_numpy(tr_pos).to(device)
    tr_m = torch.from_numpy(tr_seen).to(device)
    va_y = torch.from_numpy(va_pos).to(device)
    va_m = torch.from_numpy(va_seen).to(device)

    print(f"categories {len(categories)} | room types {len(room_types)} | embed dim {emb.shape[1]}")
    print(f"train cells {int(tr_m.sum())} ({int((tr_y * tr_m).sum())} pos) | "
          f"val cells {int(va_m.sum())} ({int((va_y * va_m).sum())} pos)")
    print(f"held-out scenes ({len(val_scenes)}): {', '.join(val_scenes)}")

    model = RelationalSemanticNetwork(emb.shape[1], len(room_types), tuple(args.hidden), args.dropout).to(device)

    n_pos = float((tr_y * tr_m).sum())
    n_neg = float(tr_m.sum()) - n_pos
    pw = args.pos_weight if args.pos_weight is not None else n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor(pw, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"ap": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(emb)
        # Mask so unobserved (category, room) cells contribute no gradient.
        loss = (loss_fn(logits, tr_y) * tr_m).sum() / tr_m.sum()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(emb)
            probs = torch.sigmoid(val_logits)
        sel = va_m.bool()
        y_np = va_y[sel].cpu().numpy()
        p_np = probs[sel].cpu().numpy()
        auc = roc_auc(y_np, p_np)
        ap = average_precision(y_np, p_np)
        brier = float(((p_np - y_np) ** 2).mean())  # comparable to SEEK Table II

        if ap > best["ap"]:
            best = {
                "ap": ap,
                "auc": auc,
                "brier": brier,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train loss {loss.item():.4f} | val AUC {auc:.4f} "
                  f"| val AP {ap:.4f} | Brier {brier:.4f}")

    print(f"\nbest epoch {best['epoch']}: val AUC {best['auc']:.4f}, "
          f"val AP {best['ap']:.4f}, Brier {best['brier']:.4f}")
    print(f"positive base rate on val: {y_np.mean():.4f}")

    # Platt scaling: logit' = a * logit + b. Fitted on the TRAINING cells only - fitting
    # on validation would make the reported Brier optimistic. Rescaling is monotonic, so
    # AUC and AP are unchanged; only calibration improves.
    calibration = {"a": 1.0, "b": 0.0}
    if args.calibrate:
        model.load_state_dict(best["state"])
        model.to(device).eval()
        with torch.no_grad():
            tr_logits = model(emb)
        sel_tr = tr_m.bool()
        zt, yt = tr_logits[sel_tr].detach(), tr_y[sel_tr]
        a = torch.ones(1, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        cal_opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            cal_opt.zero_grad()
            loss_c = bce(a * zt + b, yt)
            loss_c.backward()
            return loss_c

        cal_opt.step(closure)
        calibration = {"a": float(a.item()), "b": float(b.item())}

        with torch.no_grad():
            cal_probs = torch.sigmoid(a * model(emb) + b)
        cp = cal_probs[va_m.bool()].cpu().numpy()
        print(f"calibration: logit' = {calibration['a']:.3f} * logit + {calibration['b']:.3f}")
        print(f"  mean predicted {cp.mean():.4f} vs base rate {y_np.mean():.4f} "
              f"| Brier {float(((cp - y_np) ** 2).mean()):.4f} "
              f"(AUC/AP unchanged: {roc_auc(y_np, cp):.4f}/{average_precision(y_np, cp):.4f})")
        best["brier"] = float(((cp - y_np) ** 2).mean())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": best["state"],
            "room_types": room_types,
            "object_categories": categories,
            "embedding_model": str(cached["model"]),
            "config": {"hidden": list(args.hidden), "dropout": args.dropout, "embed_dim": int(emb.shape[1])},
            "calibration": calibration,
            "metrics": {
                "val_auc": best["auc"],
                "val_ap": best["ap"],
                "val_brier": best["brier"],
                "epoch": best["epoch"],
            },
            "val_scenes": val_scenes,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
