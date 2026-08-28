"""Query the SEEK-style RSN: P(object present) across every room type.

Accepts ANY object name - the frozen text encoder maps unseen names into the same
space, so the model degrades gracefully instead of failing on out-of-vocabulary input.

Examples:
    python query_rsn.py --object "fire extinguisher"
    python query_rsn.py --object toaster --room kitchen
"""

import argparse

import numpy as np
import torch

from train_rsn import RelationalSemanticNetwork


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = RelationalSemanticNetwork(
        cfg["embed_dim"], len(ckpt["room_types"]), tuple(cfg["hidden"]), cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_rooms(model, ckpt, name, device):
    """Return calibrated P(name present) for every room type."""
    from embed_categories import embed_names

    vec = torch.from_numpy(embed_names([name])).to(device)
    cal = ckpt.get("calibration", {"a": 1.0, "b": 0.0})
    with torch.no_grad():
        logits = model(vec)[0]
        probs = torch.sigmoid(cal["a"] * logits + cal["b"])
    return probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/rsn_cal.pt")
    parser.add_argument("--object", required=True)
    parser.add_argument("--room")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, ckpt = load(args.model, device)
    rooms = ckpt["room_types"]

    probs = predict_rooms(model, ckpt, args.object, device)
    in_vocab = args.object.replace(" ", "_").lower() in set(ckpt["object_categories"])

    if args.room:
        room = args.room.strip().lower().replace(" ", "_").replace("-", "_")
        if room not in rooms:
            raise SystemExit(f"unknown room type: {args.room}\n  known: {', '.join(rooms)}")
        tag = "" if in_vocab else "  [name unseen in training - inferred from text embedding]"
        print(f"P({args.object} in {room}) = {probs[rooms.index(room)]:.4f}{tag}")
    else:
        tag = "" if in_vocab else " [unseen name - inferred from text embedding]"
        print(f"most likely rooms for '{args.object}'{tag}:")
        for i in np.argsort(probs)[::-1][: args.top]:
            print(f"  {rooms[i]:20s} {probs[i]:.4f}")


if __name__ == "__main__":
    main()
