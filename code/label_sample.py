"""Builds the scaffold for a genuinely unseen eval set (issue #37).

Every accuracy figure this project reports comes from `sample_messages.csv`,
which `prompt.py` was written against. #31 split it so the scored half supplies
no worked examples, but `SYSTEM` still saw all 30 rows while it was being
iterated -- so "held out" means absent from the prompt, not never seen.

The 110 rows in `messages.csv` are the only genuinely unseen data here. This
module picks 20 of them and writes an empty label sheet for a human to fill in.

    python label_sample.py        # -> dataset/messages_labelled.csv

It deliberately does not fill in the labels. They have to come from a person:
labels produced by the same model and rubric under test would make the score a
self-consistency measurement wearing the costume of an accuracy measurement.

Two things to know about how the 20 are chosen.

**It stratifies on predictions, not truth**, because no gold labels exist to
stratify against. That biases which rows get labelled toward covering the
prediction space -- a class the router never predicts cannot be sampled for. It
does not bias the labels a human then assigns.

**It is coverage-first, not distribution-matched.** The predictions occupy 15
distinct (action, type) cells and there are only 20 slots, so each cell gets
roughly one row. Real traffic is nothing like that flat: `mute/scam` alone is 31
of 110 and would take 5.6 slots proportionally, but takes 1 here. The resulting
score therefore answers "does the rubric hold up across every class" and NOT
"what is accuracy on realistic traffic" -- do not compare it directly against
the `sample_messages.csv` figures, which are distribution-matched to nothing in
particular either. If you want a proportional sample instead, weight `order` by
cell size rather than visiting each cell once per round.
"""
import csv

from common import DATA, load

SAMPLE_SIZE = 20
SHEET = DATA / "messages_labelled.csv"
LABEL_COLUMNS = ["action", "message_type", "reason", "confidence", "evidence_message_ids"]
# Enough of the source row to label it without opening another file.
CONTEXT_COLUMNS = [
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type", "media_id",
    "forwarded_count",
]


def stratified_ids(messages, predictions, size=SAMPLE_SIZE):
    """`size` message_ids spread across the predicted action/type cells.

    Deterministic: no RNG, no seed to remember. Cells are visited in a fixed
    order and one row is taken from each in turn until the quota is met, so a
    rare predicted class is represented before a common one is doubled up.
    """
    pred = {p["message_id"]: p for p in predictions}
    cells = {}
    for m in sorted(messages, key=lambda r: r["message_id"]):
        p = pred.get(m["message_id"])
        if not p:
            continue
        cells.setdefault((p["action"], p["message_type"]), []).append(m["message_id"])

    # Rarest cell first, so scam/spam and the other adversarial types survive
    # the quota instead of being crowded out by whatever the router predicts most.
    order = sorted(cells, key=lambda k: (len(cells[k]), k))
    picked, round_ = [], 0
    while len(picked) < size and any(len(cells[k]) > round_ for k in order):
        for key in order:
            if len(picked) >= size:
                break
            if len(cells[key]) > round_:
                picked.append(cells[key][round_])
        round_ += 1
    return picked[:size]


def write_sheet(path=SHEET):
    """Writes the label sheet. Refuses to clobber labels already entered."""
    if path.exists() and any(r.get("action") for r in csv.DictReader(open(path, encoding="utf-8"))):
        raise SystemExit(
            f"{path.name} already has labels in it. Delete it by hand if you really "
            "mean to discard them and start over."
        )
    messages = load("messages")
    ids = set(stratified_ids(messages, load("output")))
    rows = [m for m in messages if m["message_id"] in ids]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CONTEXT_COLUMNS + LABEL_COLUMNS)
        w.writeheader()
        for m in rows:
            w.writerow({**{c: m.get(c, "") for c in CONTEXT_COLUMNS}, **{c: "" for c in LABEL_COLUMNS}})
    return rows


def labelled_rows():
    """The rows of the sheet a human has actually filled in. [] if none, or no sheet."""
    if not SHEET.exists():
        return []
    return [r for r in csv.DictReader(open(SHEET, encoding="utf-8")) if (r.get("action") or "").strip()]


if __name__ == "__main__":
    written = write_sheet()
    print(f"Wrote {len(written)} unlabelled rows to {SHEET}")
    print("Fill in action, message_type, reason, confidence and evidence_message_ids by hand,")
    print("then score them with: python main.py --eval-unseen")
