"""Entry point.

  python main.py            route dataset/messages.csv -> dataset/output.csv
  python main.py --eval     route the 30 labelled sample rows and score against them
"""
import argparse
import csv
import json
import sys
from collections import Counter

import context
import media as media_mod
import route
from common import DATA, load

OUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

# The sample rows FEWSHOT is drawn from. Scoring on these measures how well the
# model reproduces cases it was shown, not whether it generalizes, so --eval
# holds them out and scores the other 15.
#
# Twelve were identified by their gold evidence ids appearing as the example's
# evidence_message_ids. The remaining three cite no evidence and were identified
# by the row's message_text appearing near-verbatim in the example's situation
# (sample_msg_049, _052, _053). test_eval_split pins the first group so that
# editing FEWSHOT cannot silently drift the split back into contamination.
FEWSHOT_SOURCE_IDS = frozenset(
    {
        "sample_msg_001", "sample_msg_002", "sample_msg_004", "sample_msg_005",
        "sample_msg_007", "sample_msg_008", "sample_msg_010", "sample_msg_014",
        "sample_msg_019", "sample_msg_043", "sample_msg_044", "sample_msg_045",
        "sample_msg_049", "sample_msg_052", "sample_msg_053",
    }
)


def eval_rows(rows, held_out_only=True):
    """The rows --eval scores.

    Held out by default. The 30 labelled rows are also the few-shot source, so a
    score over all of them is a recall number for examples already in the prompt
    and cannot go down -- which is why the headline used to read 100%.
    """
    if not held_out_only:
        return rows
    return [r for r in rows if r["message_id"] not in FEWSHOT_SOURCE_IDS]


def media_lookup(use_cache=True):
    """Ensure the media caches exist, then index them by media_id."""
    images = media_mod.describe_images(load("images"), verbose=False, use_cache=use_cache)
    voices = media_mod.transcribe_voice(load("voice_notes"), verbose=False, use_cache=use_cache)
    return {**images, **voices}


def run(rows, router, media, verbose=True):
    results = []
    for n, msg in enumerate(rows, 1):
        m = media.get(msg.get("media_id") or "")
        pack = context.build_pack(msg, m)
        decision = router.route(pack)
        results.append((msg, pack, decision))
        if verbose:
            flags = ("  [" + ",".join(decision["_rules_fired"]) + "]") if decision["_rules_fired"] else ""
            print(
                f"{n:>3}/{len(rows)} {msg['message_id']:<14} "
                f"{decision['action']:<7} {decision['message_type']:<16} "
                f"{decision['confidence']}{flags}",
                flush=True,
            )
    return results


def write_output(results, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for msg, _pack, d in results:
            w.writerow(
                {
                    "message_id": msg["message_id"],
                    "action": d["action"],
                    "message_type": d["message_type"],
                    "reason": d["reason"],
                    "confidence": d["confidence"],
                    "evidence_message_ids": d["evidence_message_ids"],
                }
            )


def validate(results, expected_ids):
    """Fail loudly rather than submit a malformed file."""
    problems = []
    got = [m["message_id"] for m, _, _ in results]
    if got != expected_ids:
        problems.append(f"row mismatch: {len(got)} rows vs {len(expected_ids)} expected")
    valid_hist = {r["message_id"] for r in load("message_history")}
    for msg, _pack, d in results:
        mid = msg["message_id"]
        if d["action"] not in route.rules.ACTIONS:
            problems.append(f"{mid}: bad action {d['action']}")
        if d["message_type"] not in route.rules.TYPES:
            problems.append(f"{mid}: bad type {d['message_type']}")
        if not 0.0 <= float(d["confidence"]) <= 1.0:
            problems.append(f"{mid}: bad confidence {d['confidence']}")
        if not d["reason"]:
            problems.append(f"{mid}: empty reason")
        ev = d["evidence_message_ids"]
        if ev != "none":
            for e in ev.split(";"):
                if e not in valid_hist:
                    problems.append(f"{mid}: unknown evidence id {e}")
    return problems


def score(results, truth_rows):
    truth = {r["message_id"]: r for r in truth_rows}
    a_hit = t_hit = both = 0
    e_hit = e_total = 0
    confusion = Counter()
    for msg, _pack, d in results:
        t = truth[msg["message_id"]]
        a_ok = d["action"] == t["action"]
        t_ok = d["message_type"] == t["message_type"]
        a_hit += a_ok
        t_hit += t_ok
        both += a_ok and t_ok
        if not a_ok:
            confusion[f"{t['action']}->{d['action']}"] += 1
        gold_ev = {x for x in t["evidence_message_ids"].split(";") if x and x != "none"}
        if gold_ev:
            e_total += 1
            mine = {x for x in d["evidence_message_ids"].split(";") if x != "none"}
            e_hit += bool(gold_ev & mine)
    n = len(results)
    print(f"\naction accuracy      {a_hit}/{n} = {a_hit / n:.0%}")
    print(f"type accuracy        {t_hit}/{n} = {t_hit / n:.0%}")
    print(f"both correct         {both}/{n} = {both / n:.0%}")
    if e_total:
        print(f"evidence overlap     {e_hit}/{e_total} = {e_hit / e_total:.0%}")
    if confusion:
        print("action errors (gold->pred):", dict(confusion))
    for msg, _pack, d in results:
        t = truth[msg["message_id"]]
        if d["action"] != t["action"] or d["message_type"] != t["message_type"]:
            print(
                f"  MISS {msg['message_id']}: gold {t['action']}/{t['message_type']} "
                f"got {d['action']}/{d['message_type']} :: {d['reason']}"
            )
    return both / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval",
        action="store_true",
        help="score against the 15 held-out rows of sample_messages.csv",
    )
    ap.add_argument(
        "--eval-all",
        action="store_true",
        help="with --eval, score all 30 rows including the few-shot source (contaminated)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=route.MODEL)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dump", default="", help="write full decision traces to this JSON path")
    args = ap.parse_args()

    print("Media pre-pass...", flush=True)
    media = media_lookup(use_cache=not args.no_cache)

    rows = load("sample_messages" if args.eval else "messages")
    if args.eval:
        rows = eval_rows(rows, held_out_only=not args.eval_all)
        if args.eval_all:
            print(
                "WARNING: scoring all 30 rows, 15 of which are the few-shot source. "
                "This number is a recall ceiling, not an accuracy estimate.",
                file=sys.stderr,
            )
    if args.limit:
        rows = rows[: args.limit]
    router = route.Router(model=args.model, use_cache=not args.no_cache)

    print(f"Routing {len(rows)} messages with {args.model}...", flush=True)
    results = run(rows, router, media)
    # Cache hits come off `asked` (messages sent), not `calls` (requests billed);
    # a retried message bills more than once and used to make this go negative.
    retried = router.calls - router.asked
    print(
        f"\n{router.asked} messages routed, {router.calls} model calls made"
        f"{f' ({retried} retried)' if retried else ''}"
        f" ({len(rows) - router.asked} served from cache)"
    )
    print("action mix:", dict(Counter(d["action"] for _, _, d in results)))
    print("type mix:  ", dict(Counter(d["message_type"] for _, _, d in results)))
    fired = Counter(f for _, _, d in results for f in d["_rules_fired"])
    if fired:
        print("rules fired:", dict(fired))

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(
                [{"pack": p, "decision": d} for _m, p, d in results], fh, indent=1, ensure_ascii=False
            )

    if args.eval:
        score(results, rows)
        return

    problems = validate(results, [r["message_id"] for r in load("messages")])
    if problems:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        sys.exit(1)
    out = DATA / "output.csv"
    write_output(results, out)
    print(f"\nWrote {len(results)} rows to {out}")


if __name__ == "__main__":
    main()
