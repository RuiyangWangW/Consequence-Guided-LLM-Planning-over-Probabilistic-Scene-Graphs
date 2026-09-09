#!/usr/bin/env python3
"""Hosted models behind the same `prompt -> text` interface the local ones use.

`planner.get_generator` hands every stage a callable taking `(prompt, max_new_tokens)` and
returning the reply as a string. Nothing above that line cares whether the weights are on the
card or on somebody else's - so an API model is a drop-in, and `gavel.decompose`, `replan.run`
and the goal adapter all work unchanged against one.

Two providers, chosen by the model id rather than a flag, because the id is what the run
records and a flag is one more thing that can disagree with it:

    gpt-*     -> OpenAI     /v1/chat/completions
    claude-*  -> Anthropic  /v1/messages

WHAT IS SENT, AND WHAT IS NOT

The instruction text and the believed scene graph. Both are public BEHAVIOR-1K benchmark
content. No credentials, no paths, nothing about the machine.

DETERMINISM, AND WHERE IT STOPS

The local runs decode greedily, so a prompt has one answer. `temperature=0` asks for the same
thing here and mostly gets it, but a hosted model is a moving target - it can be updated under
a stable id, and some reasoning models refuse the parameter outright. So an API column is not
reproducible the way a local one is, and the run records the model id and the date for exactly
that reason. Where the parameter is refused this retries without it rather than failing, and
notes it, because a run that silently sampled is worse than one that says it did.

KEYS

From the environment, or from `~/.behavior_api_keys` if the environment has none. Never logged,
never echoed, never written into a result file.
"""

import json
import os
import time

#: How many times to retry a call the provider refused for a reason that might pass later -
#: rate limits and 5xx. Anything else fails immediately, because retrying a bad request just
#: spends money slower.
RETRIES = 5

#: Seconds between attempts, doubling. The last wait is the longest a single call may stall.
BACKOFF = 2.0

#: How much thinking each provider is allowed, matched to the local runs.
#:
#: `planner._local_generator` passes `enable_thinking=False` to the Qwen3 chat template, so the
#: 4B and 8B columns do no explicit reasoning at all. Letting a hosted model reason freely would
#: stop the comparison being "which model plans better" and make it "what does a reasoning budget
#: buy" - a different question, and one that flatters the hosted column for a reason unrelated to
#: the pipeline. So each is pinned to its floor.
#:
#: The floors are not equal and the write-up should say so: OpenAI offers `none`, Anthropic's
#: lowest setting is `low`, so Claude is given slightly more thinking than Qwen or GPT and the
#: comparison is if anything generous to it.
REASONING = {"openai": "none", "anthropic": "low"}

KEYFILE = os.path.expanduser("~/.behavior_api_keys")

_SESSION = {}
_NOTED = set()

#: Tokens spent per model, so a run can be costed and a probe can extrapolate before the bill
#: is committed. Counted from what the provider reports, never estimated from the prompt.
USAGE = {}


def _record(model, usage, calls=1):
    row = USAGE.setdefault(model, {"calls": 0, "input": 0, "output": 0, "reasoning": 0})
    row["calls"] += calls
    row["input"] += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    row["output"] += usage.get("output_tokens") or usage.get("completion_tokens") or 0
    row["reasoning"] += ((usage.get("completion_tokens_details") or {})
                         .get("reasoning_tokens") or 0)


def spent():
    """What every hosted model has cost so far, in tokens."""
    return {m: dict(v) for m, v in USAGE.items()}


def _key(name):
    """The API key for `name`, from the environment or the key file. Never returned to a
    caller that did not ask for it, and never printed."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        with open(KEYFILE) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    raise RuntimeError(
        f"{name} is not set and {KEYFILE} does not carry it. Hosted models need a key; "
        f"local models do not, so this is only reached when a model id names a provider.")


def provider_of(model):
    """Which provider serves this model id, or None when it is a local model."""
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return None


def _client():
    import httpx

    if "client" not in _SESSION:
        # One connection pool for the whole run. Long prompts and 10k calls make handshakes
        # a real cost, and a fresh client per call also loses the retry state.
        _SESSION["client"] = httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0))
    return _SESSION["client"]


def _post(url, headers, payload):
    """POST with backoff on the failures that are worth retrying. Returns parsed JSON."""
    client = _client()
    wait = BACKOFF
    last = None
    for attempt in range(RETRIES):
        try:
            reply = client.post(url, headers=headers, json=payload)
        except Exception as exc:                       # connection reset, timeout, DNS
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(wait); wait *= 2
            continue
        if reply.status_code == 200:
            return reply.json()
        last = f"HTTP {reply.status_code}: {reply.text[:300]}"
        if reply.status_code in (408, 409, 429) or reply.status_code >= 500:
            # Rate limited or the provider's problem: wait and try again. Honour Retry-After
            # when it is offered, because guessing shorter than the provider asked for is how
            # a run gets itself blocked.
            hinted = reply.headers.get("retry-after")
            time.sleep(float(hinted) if hinted and hinted.isdigit() else wait)
            wait *= 2
            continue
        break                                          # 400, 401, 404: retrying cannot help
    raise RuntimeError(f"API call failed after {RETRIES} attempts - {last}")


def _openai(model, prompt, max_new_tokens, temperature, effort=None):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_completion_tokens": max_new_tokens}
    if effort:
        payload["reasoning_effort"] = effort
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}",
               "Content-Type": "application/json"}
    try:
        got = _post("https://api.openai.com/v1/chat/completions", headers, payload)
        _record(model, got.get("usage") or {})
    except RuntimeError as exc:
        # Reasoning models reject a temperature they do not honour. Retry without it and say
        # so once, rather than dropping the sample silently or failing the whole run.
        if temperature is not None and "temperature" in str(exc):
            if model not in _NOTED:
                _NOTED.add(model)
                print(f"[api] {model} refuses temperature; sampling at its own default")
            return _openai(model, prompt, max_new_tokens, None, effort)
        raise
    return (got["choices"][0]["message"]["content"] or "").strip()


def _anthropic(model, prompt, max_new_tokens, temperature, effort=None):
    payload = {"model": model, "max_tokens": max_new_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    if effort:
        # This model refuses `thinking.type.enabled` and names its own replacement in the
        # refusal: adaptive thinking, with the budget set by `output_config.effort`.
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {"x-api-key": _key("ANTHROPIC_API_KEY"),
               "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    try:
        got = _post("https://api.anthropic.com/v1/messages", headers, payload)
        _record(model, got.get("usage") or {})
    except RuntimeError as exc:
        # Same story as OpenAI: newer models deprecate the knob rather than honour it.
        if temperature is not None and "temperature" in str(exc):
            if model not in _NOTED:
                _NOTED.add(model)
                print(f"[api] {model} deprecates temperature; sampling at its own default")
            return _anthropic(model, prompt, max_new_tokens, None, effort)
        raise
    parts = [b.get("text", "") for b in got.get("content", []) if b.get("type") == "text"]
    return "".join(parts).strip()


def generator(model, temperature=0.0, effort="match-local"):
    """A `(prompt, max_new_tokens) -> text` callable for a hosted model.

    The same shape `planner._local_generator` returns, so every stage of the pipeline takes one
    without knowing the difference.
    """
    which = provider_of(model)
    if which is None:
        raise ValueError(f"{model!r} names no hosted provider; use a local generator")
    call = _openai if which == "openai" else _anthropic
    if effort == "match-local":
        effort = REASONING[which]

    def run(prompt, max_new_tokens=512):
        return call(model, prompt, int(max_new_tokens), temperature, effort)

    run.model_id = model
    run.provider = which
    run.effort = effort
    return run


def usage_probe(model, prompt="Reply with the single word: ok", max_new_tokens=64):
    """One call, timed, to check credentials and measure latency before a long run."""
    made = generator(model)
    at = time.perf_counter()
    text = made(prompt, max_new_tokens)
    return {"model": model, "provider": made.provider, "effort": made.effort,
            "seconds": round(time.perf_counter() - at, 2), "reply": text[:80]}


if __name__ == "__main__":
    import sys

    for name in (sys.argv[1:] or ["gpt-5.6-sol", "claude-sonnet-5"]):
        try:
            print(json.dumps(usage_probe(name)))
        except Exception as exc:
            print(json.dumps({"model": name, "error": f"{type(exc).__name__}: {exc}"[:200]}))
