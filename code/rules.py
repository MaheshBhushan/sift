"""Deterministic guardrails applied after the model.

Two jobs only:
  1. Hard safety floors -- things that must never be surfaced regardless of what
     the model said (impersonated domains, credential harvesting, opted-out marketing).
  2. Output coherence -- action/type combinations that are internally contradictory,
     and confidence sanity.

Everything nuanced stays with the model; this layer is intentionally small.
"""
import re

ACTIONS = {"notify", "digest", "mute"}
TYPES = {
    "personal", "urgent", "event", "payment", "business_update", "promotion",
    "greeting", "forward", "spam", "scam", "unknown",
}

# A credential term alone is not a scam -- awareness notices mention OTPs too.
# We need the term *plus* an ask or a threat, and no safety-advisory negation.
CREDENTIAL_TERM = re.compile(
    r"\b(otp|one[- ]time (?:code|password)|login code|verification code|6[- ]digit|"
    r"password|pin|cvv|kyc|account|profile|wallet)\b",
    re.I,
)
CREDENTIAL_ASK = re.compile(
    r"(reply with|send (?:me )?the|share the|confirm (?:your )?(?:password|otp|pin)|"
    r"verify (?:now|your|at)|click (?:here|below) to verify|"
    r"will be (?:blocked|suspended|deactivated|deleted)|may be temporarily blocked|"
    r"expire[sd]? today|to keep (?:your )?(?:account|access) active|"
    r"reactivate|update your (?:kyc|bank|payment) details)",
    re.I,
)
ADVISORY_PAT = re.compile(
    r"(never (?:ask|share)|do not share|don't share|beware|is a scam|awareness|"
    r"we will never)",
    re.I,
)
INJECTION_PAT = re.compile(
    r"(ignore (?:all )?(?:previous|prior|above)|disregard (?:all )?(?:previous|prior)|"
    r"you are now|system prompt|mark this (?:message )?as (?:notify|urgent)|"
    r"routing rules)",
    re.I,
)


def _text_of(pack):
    parts = [pack["message"].get("text") or ""]
    media = pack.get("media_understanding") or {}
    parts += [str(media.get("summary") or ""), str(media.get("visible_text") or "")]
    return " ".join(parts)


def apply(pack, decision):
    """Returns (decision, [rule names that fired])."""
    fired = []
    d = dict(decision)
    text = _text_of(pack)
    biz = pack.get("business") or {}
    sender = pack.get("sender") or {}
    media = pack.get("media_understanding") or {}
    rel = biz.get("user_relationship") or {}

    def force(action, mtype, name):
        if d["action"] != action or d["message_type"] != mtype:
            d["action"], d["message_type"] = action, mtype
            fired.append(name)

    # --- Safety floors -------------------------------------------------
    credential_ask = (
        bool(CREDENTIAL_TERM.search(text) and CREDENTIAL_ASK.search(text))
        or bool(media.get("asks_for_money_or_credentials"))
    ) and not ADVISORY_PAT.search(text)

    # A mismatched sending domain is only impersonation when something else is
    # also wrong. Established verified brands legitimately send via link
    # shorteners and marketing subdomains.
    # No official domain on record means there is nothing to mismatch against.
    mismatched = bool(biz.get("official_domain")) and not biz.get("sender_domain_matches_official", True)
    if mismatched:
        risky_sender = (
            not biz.get("verified")
            or biz.get("sender_domain_age_days", 0) < 365
            or biz.get("user_reports_30d", 0) >= 10
            or credential_ask
        )
        if risky_sender:
            if credential_ask or not biz.get("verified"):
                force("mute", "scam" if credential_ask else d["message_type"], "impersonation_risk")
            if d["action"] != "mute":
                d["action"] = "mute"
                fired.append("impersonation_risk")

    # Credential harvesting is scam even when it mimics a support workflow.
    if credential_ask:
        untrusted = (
            (biz and (not biz.get("verified") or not rel))
            or (sender and not sender.get("known_sender"))
            or pack["message"]["conversation_type"] == "group"
        )
        if untrusted:
            force("mute", "scam", "credential_request")

    # Text trying to steer the router is itself the signal.
    if INJECTION_PAT.search(text):
        force("mute", "scam", "prompt_injection")

    # Marketing the user explicitly switched off.
    if rel and (rel.get("promotions_opted_out_at") or rel.get("allows_promotions") is False):
        if d["message_type"] in {"promotion", "spam"} and d["action"] != "mute":
            d["action"] = "mute"
            fired.append("promotions_opted_out")

    # --- Coherence -----------------------------------------------------
    if d["message_type"] in {"scam", "spam"} and d["action"] != "mute":
        d["action"] = "mute"
        fired.append("risky_type_must_mute")

    if d["action"] not in ACTIONS:
        d["action"] = "digest"
        fired.append("invalid_action")
    if d["message_type"] not in TYPES:
        d["message_type"] = "unknown"
        fired.append("invalid_type")

    # Quiet hours soften a borderline interrupt, but never a scam call or a
    # direct personal request.
    user = pack["receiving_user"]
    group = pack.get("group") or {}
    if (
        d["action"] == "notify"
        and user.get("message_arrived_during_dnd")
        and float(d.get("confidence", 0.8)) < 0.8
        and d["message_type"] not in {"urgent", "personal"}
        and not group.get("user_directly_mentioned")
    ):
        d["action"] = "digest"
        fired.append("dnd_softening")

    # --- Confidence ----------------------------------------------------
    try:
        conf = float(d.get("confidence", 0.8))
    except (TypeError, ValueError):
        conf = 0.75
        fired.append("invalid_confidence")
    if fired and conf < 0.85:
        conf = 0.85  # a rule firing is a strong, explicit signal
    d["confidence"] = round(min(max(conf, 0.55), 0.92), 2)

    # --- Evidence hygiene ----------------------------------------------
    valid = {m["message_id"] for m in pack.get("similar_past_messages", [])}
    ids = [x.strip() for x in str(d.get("evidence_message_ids") or "").split(";")]
    ids = [x for x in ids if x and x != "none" and x in valid]
    if not ids and fired and valid:
        ids = []  # do not fabricate support for a rule-driven decision
    d["evidence_message_ids"] = ";".join(ids[:3]) if ids else "none"

    reason = str(d.get("reason") or "").strip().replace("\n", " ")
    d["reason"] = reason or "Routed on sender trust, message content, and this user's past behaviour."
    return d, fired


RULE_REASONS = {
    "impersonation_risk": "The sending domain does not match the brand and the account shows other risk signals.",
    "credential_request": "The message asks for OTP, password, or account verification through an untrusted flow.",
    "prompt_injection": "The message tries to instruct the router, so it was routed on its actual risky content.",
    "promotions_opted_out": "The user has opted out of or repeatedly dismissed similar marketing messages.",
}


def reason_override(decision, fired):
    """If a safety rule flipped the decision, say so plainly."""
    for name in ("credential_request", "prompt_injection", "impersonation_risk", "promotions_opted_out"):
        if name in fired:
            decision["reason"] = RULE_REASONS[name]
            break
    return decision
