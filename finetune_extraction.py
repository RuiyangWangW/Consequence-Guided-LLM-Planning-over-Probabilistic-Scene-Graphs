#!/usr/bin/env python3
"""LoRA fine-tune a small model for object extraction, and see whether it beats the prompt.

Prompt engineering took extraction from 56% to 82% exact on the benchmark, so the question
this answers is whether a model *trained* on the task does better than one talked into it.

Two things make the comparison worth trusting:

**The training data shares no vocabulary with the test set.** `extraction_data.py` builds
instructions from slot templates, and `--exclude-benchmark` holds back all 109 categories
the 100 benchmark tasks use. The model is trained without ever seeing the words `oven`,
`countertop` or `bottom_cabinet`, and is then asked about them. Nothing is scene-derived.

**The fine-tuned model is given a short prompt, not the long one.** Twelve worked examples
exist to demonstrate a format and an answer length; a model trained on 8,000 of them should
need neither. That makes this a fair test rather than a stacked one, and if it works the
prompt drops from ~3,300 characters to ~400.

Answers are trained in the exact format `task_objects.parse` already reads, so nothing
downstream changes.

    python finetune_extraction.py --model Qwen/Qwen3-1.7B --out models/extract-1.7b
    python finetune_extraction.py --model Qwen/Qwen3-4B   --out models/extract-4b
"""

import argparse
import json
import os
import random
import time

INSTRUCTION = """Extract the objects this household task names, and sort each one by how
much the task says about where it is. Every object goes in EXACTLY ONE of the three groups.

DEPENDENT: the task says what this object is on, in, under or next to. Write each as
"<object> <relation> <other>", relation one of INSIDE, ON_TOP, UNDER, NEXT_TO. This is
where the object is NOW, never where the task wants it to END UP: "get the potato from the
fridge" is DEPENDENT, but "put the mug in the dishwasher" is not - the mug is not in the
dishwasher yet.

STATED: the task says which room this object is in, and nothing more specific. Write each as
"<object> IN <room>". "the office bottom cabinet" and "the bottom cabinet in the office"
both mean object `bottom_cabinet`, room `office` - the room is never part of the name.

UNCERTAIN: the task names the object and says nothing about where it is.

The groups are ordered by how much they tell us, and each object takes the FIRST that
applies: a support beats a room, a room beats nothing. So an object listed in DEPENDENT is
never repeated in STATED or UNCERTAIN, and one in STATED is never repeated in UNCERTAIN.

A room is not an object. Never list a room in UNCERTAIN, and never put one on the left of IN.
Name whole objects, not parts.

Task: {task}"""


GOAL_INSTRUCTION = """State the GOAL of this household task: the conditions that must hold
when it is done. Use only these forms, one per line:

  on_top(object, surface)       the object ends resting on that surface
  inside(object, container)     the object ends inside that container
  cooked(object, true)          the object was heated in an oven, microwave or hob
  washed(object, true)          the object went through a washer or dishwasher
  dried(object, true)           the object went through a dryer

Say where each object the task moves ends up, and what was done to it. Do NOT say anything
about doors being shut or switches being off - that is checked separately, from what the
plan disturbs.

Task: {task}"""


def goal_target(row):
    """The goal, in the language `GraphMachine.unmet` tests."""
    return "\n".join(f"{k}({a}, {str(b).lower() if isinstance(b, bool) else b})"
                      for k, a, b in row["goal"])


def target(row):
    """The answer, in the format `task_objects.parse` reads.

    Three lines, one per class, written most-informative first so the model commits to a
    support before it settles for a room and to a room before it gives up. The classes are
    disjoint, which is what makes the output directly usable downstream: `populate` can
    follow each dependent object's chain to its root and take the root's room, rather than
    the caller having to work out which of two overlapping lists an object really belongs
    to.
    """
    e = row["extraction"]
    deps = ", ".join(f"{d['object']} {d['relation']} {d['target']}" for d in e["dependent"])
    stated = ", ".join(f"{o} IN {r}" for o, r in sorted(e.get("stated", {}).items()))
    return (f"DEPENDENT: {deps}\n"
            f"STATED: {stated}\n"
            f"UNCERTAIN: {', '.join(e['uncertain'])}")


def encode(rows, tok, max_len=512, kind="extraction"):
    """Tokenize into (ids, labels), with the prompt masked out of the loss.

    Training on the prompt tokens as well would spend most of the gradient teaching the
    model to reproduce an instruction it is always given.
    """
    import torch

    prompt_of = (GOAL_INSTRUCTION if kind == "goal" else INSTRUCTION)
    answer_of = (goal_target if kind == "goal" else target)

    out = []
    for row in rows:
        messages = [{"role": "user", "content": prompt_of.format(task=row["task"])}]
        try:
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        except (TypeError, ValueError):
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer_of(row) + tok.eos_token,
                    add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        out.append((torch.tensor(ids), torch.tensor(labels)))
    return out


def batches(data, size, pad_id, shuffle=True, seed=0):
    """Length-sorted batches, so padding is not most of the compute."""
    import torch

    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    if shuffle:
        random.Random(seed).shuffle(groups)
    for group in groups:
        items = [data[i] for i in group]
        width = max(len(ids) for ids, _ in items)
        ids = torch.full((len(items), width), pad_id, dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for r, (i, l) in enumerate(items):
            ids[r, :len(i)], labels[r, :len(l)], mask[r, :len(i)] = i, l, 1
        yield ids, labels, mask


def evaluate(model, data, pad_id, size):
    """Mean loss on held-out data - a training monitor, not the headline metric.

    The number that matters is extraction accuracy on the benchmark, which
    `extraction_eval.py` measures against categories this model was never trained on.
    """
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ids, labels, mask in batches(data, size, pad_id, shuffle=False):
            ids, labels, mask = (t.to(model.device) for t in (ids, labels, mask))
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            total += loss.item() * len(ids)
            count += len(ids)
    model.train()
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train", default="data/extraction-train.json")
    parser.add_argument("--val", default="data/extraction-val.json")
    parser.add_argument("--out", default="models/extract-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--limit", type=int, help="train on this many examples only")
    parser.add_argument("--target", choices=("extraction", "goal"), default="extraction",
                        help="what to learn: the objects a task names, or the state it "
                             "should end in")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train_rows = json.load(open(args.train))
    if args.limit:
        train_rows = train_rows[:args.limit]
    val_rows = json.load(open(args.val))
    train = encode(train_rows, tok, kind=args.target)
    val = encode(val_rows, tok, kind=args.target)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False
    # Attention *and* MLP projections. Attention-only adapters are cheaper but this task
    # is largely lexical - which words are objects - and that lives in the MLPs.
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: training {trainable/1e6:.1f}M of {total/1e9:.2f}B parameters "
          f"({trainable/total:.2%})")
    print(f"{len(train)} training examples, {len(val)} validation")

    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    total_steps = max(1, int(steps_per_epoch * args.epochs) // args.accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total_steps, pct_start=0.03)

    print(f"\nbaseline val loss {evaluate(model, val, pad_id, args.batch):.4f}")
    started, step, done, running = time.time(), 0, 0, 0.0
    model.train()
    for epoch in range(int(args.epochs) + 1):
        if done >= total_steps:
            break
        for i, (ids, labels, mask) in enumerate(
                batches(train, args.batch, pad_id, seed=epoch)):
            ids, labels, mask = (t.to(model.device) for t in (ids, labels, mask))
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / args.accum).backward()
            running += loss.item()
            step += 1
            if step % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            if done % 50 == 0:
                mins = (time.time() - started) / 60
                print(f"  step {done:4d}/{total_steps}  loss {running/(50*args.accum):.4f}"
                      f"  lr {sched.get_last_lr()[0]:.2e}  {mins:.1f} min", flush=True)
                running = 0.0
            if done >= total_steps:
                break

    val_loss = evaluate(model, val, pad_id, args.batch)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "training.json"), "w") as f:
        json.dump({"base": args.model, "target": args.target,
                   "examples": len(train), "epochs": args.epochs,
                   "lr": args.lr, "rank": args.rank, "val_loss": val_loss,
                   "minutes": (time.time() - started) / 60}, f, indent=1)
    print(f"\nfinal val loss {val_loss:.4f} after {(time.time()-started)/60:.1f} min")
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
