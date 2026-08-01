"""Media understanding pre-pass.

Images  -> one Claude vision call each, producing a structured summary the
           router can reason over as if it were text.
Voice   -> faster-whisper transcription (Claude has no audio input).

Both are cached to cache/*.json, so the pre-pass is paid exactly once.
"""
import base64
import json
import shutil
import subprocess
import sys

import anthropic

from common import DATA, JsonCache, api_key

VISION_MODEL = "claude-sonnet-5"

IMAGE_PROMPT = """You are reading an image attached to a WhatsApp message, so a \
notification router can decide whether to interrupt the user.

Return ONLY JSON with these keys:
- "kind": one of poster, screenshot, photo, document, payment_request, flyer, other
- "summary": 1-2 sentences describing what the image actually says/shows. Quote key text \
(offers, amounts, dates, deadlines, phone numbers, URLs) verbatim.
- "visible_text": the important text in the image, condensed
- "has_offer_or_discount": true/false
- "asks_for_money_or_credentials": true/false (payment link, UPI/QR, OTP, login, "verify account")
- "has_deadline_or_time_pressure": true/false
- "brand_claimed": brand or organisation the image claims to be from, or null
- "risk_cues": list of concrete scam/phishing cues you can see (lookalike domain, \
prize claim, urgency + payment, mismatched branding, poor spelling). Empty list if none.

Be literal. Do not speculate beyond what is visible."""

VOICE_PROMPT_NOTE = "Transcript of a WhatsApp voice note."


def _client():
    return anthropic.Anthropic(api_key=api_key())


def _prepare(path):
    """(media_type, bytes) for the API. Extensions in this dataset lie: .jpg files
    are variously PNG, WebP or AVIF, and AVIF is not an accepted input format."""
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", data
    if data[:3] == b"GIF":
        return "image/gif", data
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", data
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg", data
    # Anything else (AVIF/HEIC) -- transcode to PNG with ffmpeg. Not a rare branch:
    # img_020.jpg is AVIF, so every full run lands here at least once.
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            f"{path.name} is not a format the API accepts and needs transcoding, "
            "but ffmpeg is not installed. See the prerequisites in code/README.md."
        )
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "image2", "-c:v", "png", "-"],
        capture_output=True,
        check=True,
    )
    return "image/png", out.stdout


def _usable(entry):
    """A cached description worth reusing.

    A degraded entry is a parse failure that was written to disk so the current
    run had something to work with, not an answer. Treating it as a hit made the
    failure permanent: the image kept its empty risk_cues on every later run and
    nothing ever re-asked.
    """
    return bool(entry) and not entry.get("degraded")


def describe_images(image_rows, verbose=True, use_cache=True):
    cache = JsonCache("images.json")
    client = None
    for row in image_rows:
        iid = row["image_id"]
        if use_cache and _usable(cache.get(iid)):
            continue
        client = client or _client()
        media_type, raw_bytes = _prepare(DATA / row["file_path"])
        b64 = base64.standard_b64encode(raw_bytes).decode()
        resp = client.messages.create(
            model=VISION_MODEL,
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                }
            ],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        try:
            parsed = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            # The stub carries none of the risk fields the prompt asks for, so the
            # router scores this image as though it showed nothing alarming. Mark
            # it so the next run re-asks instead of inheriting that silently.
            parsed = {"kind": "other", "summary": raw[:400], "risk_cues": [], "degraded": True}
            print(
                f"  image {iid}: vision response did not parse; cached without risk "
                f"cues and will be retried on the next run",
                file=sys.stderr,
            )
        parsed["modality"] = "image"
        cache.set(iid, parsed)
        if verbose:
            print(f"  image {iid}: {parsed.get('kind')} - {parsed.get('summary','')[:80]}")
    return cache.data


def transcribe_voice(voice_rows, verbose=True):
    cache = JsonCache("audio.json")
    todo = [r for r in voice_rows if not cache.get(r["voice_note_id"])]
    if not todo:
        return cache.data
    from faster_whisper import WhisperModel  # imported lazily: slow, optional

    model = WhisperModel("base", device="cpu", compute_type="int8")
    for row in todo:
        vid = row["voice_note_id"]
        segments, info = model.transcribe(str(DATA / row["file_path"]), beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        entry = {
            "modality": "voice",
            "kind": "voice_note",
            "language": info.language,
            "duration_seconds": round(info.duration, 1),
            "transcript": text,
            "summary": f"{VOICE_PROMPT_NOTE} ({round(info.duration)}s): {text}",
        }
        cache.set(vid, entry)
        if verbose:
            print(f"  voice {vid} ({entry['duration_seconds']}s): {text[:80]}")
    return cache.data


if __name__ == "__main__":
    from common import load

    print("Images...")
    describe_images(load("images"))
    print("Voice notes...")
    transcribe_voice(load("voice_notes"))
    print("Media pre-pass complete.", file=sys.stderr)
