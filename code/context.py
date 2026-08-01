"""Builds a compact, personalized context pack for one incoming message.

Everything the router needs to make a decision -- and nothing else, so the
prompt stays small and the model stays focused.
"""
from difflib import SequenceMatcher

from common import group_by, i, index, load

USERS = index(load("users"), "user_id")
GROUPS = index(load("groups"), "group_id")
MEMBERS = index(load("group_members"), "group_id", "user_id")
BUSINESS = index(load("business_accounts"), "business_id")
UBH = index(load("user_business_history"), "user_id", "business_id")
HISTORY = load("message_history")
HIST_BY_USER = group_by(HISTORY, "user_id")
EVENTS = index(load("message_events"), "user_id", "message_id")
DAILY = group_by(load("daily_notification_summary"), "user_id")
IMAGES = {r["image_id"]: r["file_path"] for r in load("images")}
VOICE = {r["voice_note_id"]: r["file_path"] for r in load("voice_notes")}

MAX_EVIDENCE = 8


def _sim(a, b):
    a, b = (a or "").lower()[:400], (b or "").lower()[:400]
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _same_conversation(msg, hist):
    for key in ("group_id", "business_id", "sender_user_id"):
        if msg.get(key) and msg[key] == hist.get(key):
            return True
    return False


def _outcome(user_id, message_id):
    """How this user actually reacted to a past message -- the strongest
    personalization signal in the dataset."""
    ev = EVENTS.get((user_id, message_id))
    if not ev:
        return None
    tags = []
    if i(ev, "message_opened"):
        tags.append("opened")
    else:
        tags.append("not_opened")
    if i(ev, "message_replied"):
        tags.append(f"replied_in_{ev.get('reaction_time_minutes') or '?'}min")
    if i(ev, "notification_dismissed"):
        tags.append("dismissed")
    if i(ev, "muted_after_message"):
        tags.append("MUTED_AFTER")
    if i(ev, "message_reported"):
        tags.append("REPORTED")
    return ",".join(tags)


def retrieve(msg, media_text=""):
    """Rank this user's past messages by conversation match + text similarity."""
    text = (msg.get("message_text") or "") + " " + media_text
    scored = []
    for h in HIST_BY_USER.get(msg["user_id"], []):
        same = _same_conversation(msg, h)
        same_sender = bool(msg.get("sender_user_id")) and msg["sender_user_id"] == h.get("sender_user_id")
        sim = _sim(text, h.get("message_text"))
        # Same person in the same conversation is the strongest evidence link,
        # and survives the fact that media messages carry little text of their own.
        score = (2.0 if same else 0.0) + (1.0 if same_sender else 0.0) + 3.0 * sim
        if score <= 0.35:
            continue
        scored.append((score, sim, same, h))
    scored.sort(key=lambda t: -t[0])
    out = []
    for score, sim, same, h in scored[:MAX_EVIDENCE]:
        out.append(
            {
                "message_id": h["message_id"],
                "created_at": h["created_at"],
                "same_conversation": same,
                "text_similarity": round(sim, 2),
                "media_type": h.get("media_type") or "",
                "forwarded_count": i(h, "forwarded_count"),
                "text": (h.get("message_text") or "")[:300],
                "user_reaction": _outcome(msg["user_id"], h["message_id"]),
            }
        )
    return out


def _in_dnd(created_at, window):
    """created_at is 'YYYY-MM-DD HH:MM'; window is 'HH:MM-HH:MM', may wrap."""
    if not window or "-" not in window or " " not in created_at:
        return False
    hhmm = created_at.split(" ")[1][:5]
    start, end = window.split("-")
    return (start <= hhmm or hhmm < end) if start > end else (start <= hhmm < end)


def _sender_profile(msg):
    """What we know about a personal sender: have they ever appeared before?"""
    sender = msg.get("sender_user_id")
    if not sender:
        return None
    prior = [h for h in HIST_BY_USER.get(msg["user_id"], []) if h.get("sender_user_id") == sender]
    reactions = [_outcome(msg["user_id"], h["message_id"]) for h in prior]
    reactions = [r for r in reactions if r]
    return {
        "sender_user_id": sender,
        "prior_messages_to_user": len(prior),
        "known_sender": bool(prior),
        "past_reactions": reactions,
    }


def _group_profile(msg):
    gid = msg.get("group_id")
    if not gid:
        return None
    g = GROUPS.get(gid, {})
    mem = MEMBERS.get((gid, msg["user_id"]))
    sender_mem = MEMBERS.get((gid, msg.get("sender_user_id") or ""))
    return {
        "group_name": g.get("group_name"),
        "group_type": g.get("group_type"),
        "member_count": i(g, "member_count"),
        "messages_30d": i(g, "messages_30d"),
        "user_role": (mem or {}).get("role"),
        "user_muted_this_group": bool(i(mem, "group_muted_by_user")),
        "user_messages_read_30d": i(mem, "messages_read_30d"),
        "user_replies_30d": i(mem, "replies_sent_30d"),
        "user_dismissals_30d": i(mem, "notifications_dismissed_30d"),
        "sender_role_in_group": (sender_mem or {}).get("role"),
        "user_directly_mentioned": f"@{msg['user_id']}" in (msg.get("message_text") or ""),
    }


def _business_profile(msg):
    bid = msg.get("business_id")
    if not bid:
        return None
    b = BUSINESS.get(bid, {})
    rel = UBH.get((msg["user_id"], bid))
    domain_ok = (b.get("domain_used_by_sender") or "") == (b.get("official_domain") or "")
    return {
        "display_name": b.get("display_name"),
        "brand_name": b.get("brand_name"),
        "category": b.get("category"),
        "verified": bool(i(b, "verified")),
        "official_domain": b.get("official_domain"),
        "domain_used_by_sender": b.get("domain_used_by_sender"),
        "sender_domain_matches_official": domain_ok,
        "sender_domain_age_days": i(b, "domain_used_by_sender_age_days"),
        "account_age_days": i(b, "account_age_days"),
        "messages_sent_30d": i(b, "messages_sent_30d"),
        "user_reports_30d": i(b, "user_reports_30d"),
        "user_relationship": None
        if not rel
        else {
            "why_user_knows_account": rel.get("why_user_knows_account"),
            "last_activity_at": rel.get("last_activity_at"),
            "allows_promotions": bool(i(rel, "allows_promotions")),
            "promotions_opted_out_at": rel.get("promotions_opted_out_at") or None,
            "activity_count_180d": i(rel, "activity_count_180d"),
            "messages_opened_30d": i(rel, "messages_opened_30d"),
            "messages_dismissed_30d": i(rel, "messages_dismissed_30d"),
            "messages_replied_30d": i(rel, "messages_replied_30d"),
        },
    }


def _load_profile(user_id):
    rows = sorted(DAILY.get(user_id, []), key=lambda r: r["date"])[-14:]
    if not rows:
        return None
    sent = sum(i(r, "notifications_sent") for r in rows)
    dismissed = sum(i(r, "notifications_dismissed") for r in rows)
    return {
        "avg_notifications_per_day": round(sent / len(rows), 1),
        "dismissal_rate": round(dismissed / sent, 2) if sent else 0.0,
    }


def build_pack(msg, media=None):
    """media: {'kind': 'image'|'voice', 'summary': str, ...} or None."""
    u = USERS.get(msg["user_id"], {})
    media_text = (media or {}).get("summary", "")
    pack = {
        "message": {
            "message_id": msg["message_id"],
            "conversation_type": msg["conversation_type"],
            "created_at": msg["created_at"],
            "text": msg.get("message_text") or "",
            "media_type": msg.get("media_type") or "none",
            "forwarded_count": i(msg, "forwarded_count"),
        },
        "media_understanding": media,
        "receiving_user": {
            "user_id": msg["user_id"],
            "do_not_disturb_window": u.get("do_not_disturb_window"),
            "message_arrived_during_dnd": _in_dnd(msg["created_at"], u.get("do_not_disturb_window", "")),
            "messages_opened_30d": i(u, "messages_opened_30d"),
            "messages_replied_30d": i(u, "messages_replied_30d"),
            "notifications_dismissed_30d": i(u, "notifications_dismissed_30d"),
            "messages_reported_30d": i(u, "messages_reported_30d"),
            "notification_load": _load_profile(msg["user_id"]),
        },
        "group": _group_profile(msg),
        "business": _business_profile(msg),
        "sender": _sender_profile(msg),
        "similar_past_messages": retrieve(msg, media_text),
    }
    return {k: v for k, v in pack.items() if v is not None}
