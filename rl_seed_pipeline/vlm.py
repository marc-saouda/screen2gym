"""Vision-language inference stage.

Takes the deterministic *evidence pack* (event summary, action digest,
reconstructed typed text, visited URLs) plus a handful of labeled key-frame
screenshots, and asks a vision-capable LLM to emit the final RL-seed object:

    { "task": {...}, "initial_state": {...} }

The call uses the OpenAI-compatible Chat Completions API (works with OpenAI,
Azure OpenAI, or any compatible gateway). If no API key/credentials are
available the stage degrades gracefully: it returns ``None`` and the caller
writes the evidence pack so the inference can be run later (or fed to any VLM).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import media


def load_env(root_env_path: Optional[str] = None) -> None:
    """Populate OPENAI_API_KEY (and friends) from a .env file if not set.

    Precedence (an existing process env var is NEVER overwritten):
      1. Current process environment.
      2. python-dotenv search from the cwd upwards.
      3. An explicit ``root_env_path`` if the caller provides one.
      4. ``<repo>/../.env`` (workspace root) or ``<repo>/.env`` (repo-relative).
    Falls back to a tiny built-in parser if python-dotenv is unavailable.
    No machine-specific absolute paths are hard-coded, so this is portable.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        root_env_path,                              # optional explicit override
        os.path.join(here, "..", ".env"),           # workspace root (repo parent)
        os.path.join(here, ".env"),                 # alongside the pipeline
    ]
    candidates = [c for c in candidates if c]

    try:
        from dotenv import load_dotenv, find_dotenv
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
        for path in candidates:
            if os.environ.get("OPENAI_API_KEY"):
                break
            if path and os.path.exists(path):
                load_dotenv(path, override=False)
    except Exception:
        pass

    if os.environ.get("OPENAI_API_KEY"):
        return
    # Manual fallback parser (no dependency on python-dotenv).
    for path in candidates:
        if os.environ.get("OPENAI_API_KEY"):
            break
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):]
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(),
                                          val.strip().strip('"').strip("'"))
        except Exception:
            pass


# Attempt to load credentials as soon as the module is imported.
load_env()

# ---------------------------------------------------------------------------
# Output schema (documented for both the model and downstream consumers).
# ---------------------------------------------------------------------------
SEED_SCHEMA = {
    "task": {
        "title": "short title of what the user was doing",
        "summary": "2-4 sentence description of the end-to-end task",
        "goal": "the concrete objective / definition of done",
        "domain": "e.g. business-automation, data-entry, web-research",
        "applications": ["apps/services involved"],
        "steps": ["ordered high-level steps the user took"],
        "success_criteria": ["objectively checkable conditions for completion"],
    },
    "initial_state": {
        "narrative": "what exists on screen and in the environment at t=0, "
                     "i.e. the state every RL episode resets to",
        "starting_screen": "what the agent sees first",
        "environment": {
            "os": "operating system",
            "applications_available": ["installed apps / open tabs"],
            "accounts_and_auth": ["accounts that must be logged-in / credentials"],
        },
        "resources": [
            {
                "name": "resource name",
                "type": "file | spreadsheet | slack_channel | account | webapp",
                "location": "path or URL",
                "state": "present | absent | empty | to_be_created",
                "details": "anything needed to recreate it",
            }
        ],
        "seed_data": {
            "name": "logical name of the dataset",
            "source": "where it lives at t=0",
            "schema": ["column/field names"],
            "rows": ["the actual records needed to seed the episode"],
        },
        "preconditions": ["facts that must hold before the episode starts"],
        "reset_instructions": ["how to programmatically reset to this state"],
    },
}

SYSTEM_PROMPT = """\
You are an expert at analyzing screen recordings of computer work in order to \
seed reinforcement-learning (RL) environments for computer-use agents.

You are given:
  - a deterministic digest of the recording's input events (mouse, keyboard, \
typed text, active windows, visited URLs), and
  - a set of labeled key-frame screenshots sampled across the session.

Produce a SINGLE JSON object with exactly two top-level keys: "task" and \
"initial_state".

Definitions you MUST follow:
  - "task" = WHAT the user was trying to accomplish, end to end. Include an \
objectively checkable success criteria list.
  - "initial_state" = the STARTING STATE that each RL episode must reset to so \
an agent could attempt the task from scratch. This is the most important part. \
It must capture: the starting screen, the apps/accounts that must already be \
available and logged in, every resource that exists at t=0 (and which ones must \
be created during the task vs. pre-existing), and any SEED DATA (e.g. the exact \
rows/records the workflow operates on). Distinguish clearly between what is \
GIVEN at the start and what the agent must PRODUCE.

Rules:
  - Ground every claim in the provided evidence and screenshots. Do not invent \
URLs, account names, or data you cannot see.
  - SEED DATA IS CRITICAL: when a data table is visible anywhere (e.g. the \
source spreadsheet in WPS Office, or the populated Google Sheet), transcribe \
EVERY row verbatim into initial_state.seed_data.rows. Do NOT sample, summarize, \
truncate, or stop at a few examples -- include all rows you can read across all \
frames, de-duplicated. Represent each row as a JSON object keyed by the column \
names. Prefer the original source document for spelling and value formats.
  - Distinguish what is GIVEN at t=0 from what the agent must PRODUCE, and set \
each resource's "state" accordingly (present | absent | empty | to_be_created).
  - Prefer specific, reproducible facts over vague descriptions.
  - Output ONLY the JSON object, no prose, no code fences.
"""


def build_user_prompt(evidence: dict) -> str:
    """Assemble the text portion of the prompt from the evidence pack."""
    parts = []
    parts.append("# SESSION SUMMARY\n" + json.dumps(evidence["summary"], indent=2))

    if evidence.get("brief_text"):
        parts.append("# ON-SCREEN TASK BRIEF (read from the first frame)\n"
                     + evidence["brief_text"])

    parts.append("# VISITED URLS (ordered)\n" + "\n".join(
        f'{u["t"]}s  {u["url"]}' for u in evidence["visited_urls"][:60]))

    parts.append("# RECONSTRUCTED TYPED TEXT (ordered)\n" + "\n".join(
        f'[{t["t"]}s {t["window"]}] {t["text"]!r}'
        for t in evidence["typed_texts"]))

    parts.append("# ACTION DIGEST (compact transcript)\n" + evidence["action_digest"])

    parts.append(
        "# TARGET OUTPUT SCHEMA (fill this shape; values are descriptions)\n"
        + json.dumps(SEED_SCHEMA, indent=2))

    parts.append(
        "Now output the JSON object with keys 'task' and 'initial_state'. "
        "The attached images are the labeled key-frames referenced by the "
        "digest; use them to read exact data, app states, and the initial "
        "screen.")
    return "\n\n".join(parts)


def _build_messages(evidence: dict, keyframes: list, max_images: int = 16):
    content = [{"type": "text", "text": build_user_prompt(evidence)}]
    attached = 0
    for kf in keyframes:
        if attached >= max_images:
            break
        path = kf.get("path") if isinstance(kf, dict) else getattr(kf, "path", None)
        if not path or not os.path.exists(path):
            continue
        b64 = media.encode_image_b64(path)
        if not b64:
            continue
        label = kf.get("label") if isinstance(kf, dict) else kf.label
        offset = kf.get("offset") if isinstance(kf, dict) else kf.offset
        content.append({"type": "text", "text": f"[key-frame @ {offset}s — {label}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        attached += 1
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ], attached


def have_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or
               (os.environ.get("AZURE_OPENAI_API_KEY") and
                os.environ.get("AZURE_OPENAI_ENDPOINT")))


def infer_seed(evidence: dict, keyframes: list,
               model: str = "gpt-4o", max_images: int = 16) -> Optional[dict]:
    """Run the VLM to produce the seed object. Returns None if no credentials."""
    if not have_credentials():
        return None
    try:
        from openai import OpenAI, AzureOpenAI
        if os.environ.get("AZURE_OPENAI_API_KEY"):
            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION",
                                           "2024-08-01-preview"),
            )
        else:
            client = OpenAI()
        messages, n = _build_messages(evidence, keyframes, max_images)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=4000,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:  # pragma: no cover - network/credential dependent
        print(f"[vlm] inference failed: {e}")
        return None
