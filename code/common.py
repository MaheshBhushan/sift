"""Shared loading helpers. Stdlib only -- the dataset is small enough that
dicts beat pandas here."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)


def load(name):
    with open(DATA / f"{name}.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def index(rows, *keys):
    """Index rows by one key (dict) or a key tuple."""
    if len(keys) == 1:
        return {r[keys[0]]: r for r in rows}
    return {tuple(r[k] for k in keys): r for r in rows}


def group_by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def i(row, key, default=0):
    """Ints out of CSV strings, which are frequently blank."""
    if not row:
        return default
    v = (row.get(key) or "").strip()
    try:
        return int(v)
    except ValueError:
        return default


def api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("ANTHROPIC_API_KEY not set (env var or .env file)")


class JsonCache:
    """Dumb file-backed cache so reruns never re-pay for media or routing."""

    def __init__(self, name):
        self.path = CACHE / name
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=1))
