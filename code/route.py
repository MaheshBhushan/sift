"""One model call per message, cached, with the safety pass applied after."""
import json
import time

import anthropic

import prompt
import rules
from common import API_MAX_RETRIES, JsonCache, api_key

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1200
# A malformed response is usually the same pack hitting the same ceiling, not bad
# luck: a prompt that overruns max_tokens truncates identically every time. Give
# it one retry, so a systematic format regression costs 2 calls per message
# rather than a full ladder of them.
BAD_JSON_ATTEMPTS = 2


class MalformedResponse(ValueError):
    """The call succeeded but the body was not usable JSON.

    Its own type, so the retry loop cannot also swallow an unrelated ValueError
    raised while building the request -- that is a bug to surface, not to retry.
    Subclasses ValueError so existing callers catching ValueError still work.
    """


# Malformed JSON in an otherwise well-formed response -- the SDK cannot see these,
# so they are ours to retry.
BAD_JSON = (json.JSONDecodeError, MalformedResponse, IndexError)


def _text(resp):
    """Last text block -- responses may lead with thinking blocks."""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


class Router:
    def __init__(self, model=MODEL, use_cache=True):
        self.client = anthropic.Anthropic(api_key=api_key(), max_retries=API_MAX_RETRIES)
        self.model = model
        # Keyed on the rubric as well as the model: a decision cached under an
        # older prompt is not a valid answer for the current one.
        self.cache = JsonCache(f"routing_{model}_{prompt.fingerprint()}.json") if use_cache else None
        self.calls = 0
        # Messages that reached the model at all. `calls` counts billed requests
        # and a retried message bills several, so the two diverge -- and only
        # this one can be subtracted from the row count to get cache hits.
        self.asked = 0

    def _ask(self, pack):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": prompt.SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt.user_message(pack)}],
        )
        # Counts billed requests, so it sits after the response arrives and
        # before the body is judged: an unparseable answer was still paid for.
        # Retries therefore push this above the number of messages routed, which
        # is why `asked` exists separately.
        self.calls += 1
        raw = _text(resp)
        if "{" not in raw or "}" not in raw:
            raise MalformedResponse(
                f"no JSON in response (stop_reason={resp.stop_reason}): {raw[:200]!r}"
            )
        return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])

    def _ask_retrying(self, pack):
        """Retries a malformed response, and only that.

        Transport failures are the SDK's job (API_MAX_RETRIES), so this loop no
        longer catches them. It used to, on top of the SDK's own retries, which
        multiplied out to as many as twelve attempts for one message on a
        sustained rate limit while MAX_ATTEMPTS = 4 sat in the source implying
        four.

        A malformed response is worth exactly one more try: the same pack
        overruns max_tokens the same way every time.
        """
        for attempt in range(BAD_JSON_ATTEMPTS):
            try:
                return self._ask(pack)
            except BAD_JSON:
                if attempt == BAD_JSON_ATTEMPTS - 1:
                    raise
                time.sleep(2**attempt)

    def route(self, pack):
        mid = pack["message"]["message_id"]
        cached = self.cache.get(mid) if self.cache else None
        if cached:
            raw = cached
        else:
            self.asked += 1
            raw = self._ask_retrying(pack)
            if self.cache:
                self.cache.set(mid, raw)

        decision, fired = rules.apply(pack, raw)
        decision = rules.reason_override(decision, fired)
        decision["_rules_fired"] = fired
        decision["_model_action"] = raw.get("action")
        decision["_model_type"] = raw.get("message_type")
        return decision
