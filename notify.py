import os
import requests

EMOJI = {"fit": "🟢", "borderline": "🟡", "not fit": "🔴"}

def notify(paused: dict, thread_id: str) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")                       # blank 1: which env variable?
    if not url:
        return                                   # no webhook set up: skip quietly

    text = f"{EMOJI.get(paused['verdict'], '⚪')} **{paused['verdict']}**: {paused['job']} (score {paused['score']})"
    if paused.get("contains_injection"):                             # blank 2: which key flags an injection?
        text += "\n⚠️ Posting contains text that looks like instructions to AI tools"
    text += f"\nReview ticket: `{thread_id}`"

    try:
        requests.post(url, json={"content": text}, timeout=5)
    except requests.RequestException:
        pass                                     # a failed ping must never break the assessment