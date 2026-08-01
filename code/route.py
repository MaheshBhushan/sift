"""One model call per message, cached, with the safety pass applied after."""
import json

import anthropic

import prompt
import rules
from common import JsonCache, api_key

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1200


def _text(resp):
    """Last text block -- responses may lead with thinking blocks."""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


class Router:
    def __init__(self, model=MODEL, use_cache=True):
        self.client = anthropic.Anthropic(api_key=api_key())
        self.model = model
        self.cache = JsonCache(f"routing_{model}.json") if use_cache else None
        self.calls = 0

    def _ask(self, pack):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": prompt.SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt.user_message(pack)}],
        )
        self.calls += 1
        raw = _text(resp)
        if "{" not in raw or "}" not in raw:
            raise ValueError(f"no JSON in response (stop_reason={resp.stop_reason}): {raw[:200]!r}")
        return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])

    def route(self, pack):
        mid = pack["message"]["message_id"]
        cached = self.cache.get(mid) if self.cache else None
        if cached:
            raw = cached
        else:
            try:
                raw = self._ask(pack)
            except (json.JSONDecodeError, ValueError, IndexError):
                raw = self._ask(pack)  # one retry: transient bad JSON
            if self.cache:
                self.cache.set(mid, raw)

        decision, fired = rules.apply(pack, raw)
        decision = rules.reason_override(decision, fired)
        decision["_rules_fired"] = fired
        decision["_model_action"] = raw.get("action")
        decision["_model_type"] = raw.get("message_type")
        return decision
