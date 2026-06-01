"""Episode segmentation: split one recording into a curriculum of sub-tasks.

A long recording (here ~1h) is decomposed into a sequence of smaller,
independently-resettable RL episodes (e.g. "create + populate the sheet",
"create the Slack channel", "build the Zap trigger", ... "validate"). Each
episode carries:

* ``id`` / ``title`` / ``goal`` / ``success_criteria`` — human-readable spec,
* ``app`` — the primary web app the episode happens in,
* ``time_range`` — the [start, end] offsets in the source recording,
* ``initial_overrides`` — the cumulative environment state the episode resets
  to (what already exists at its start), expressed with a small set of knobs,
* ``checks`` / ``reward_weights`` — the grader spec the RL env consumes.

The structure is derived deterministically from the event timeline (app +
URL-group phases, plus typed-text keywords). An optional VLM pass can rewrite
the human-readable labels; the deterministic path always works offline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from events import EventTimeline


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------
def url_group(url: Optional[str]) -> Optional[str]:
    """Coarse activity group for a browser URL."""
    if not url:
        return None
    u = url.lower()
    if "docs.google.com/spreadsheets" in u:
        return "google_sheets"
    if "app.slack.com" in u or ".slack.com" in u or "//slack.com" in u:
        return "slack"
    if "zapier.com" in u:
        if "/published" in u:
            return "zapier_publish"
        if "/run/" in u or "run-details" in u:
            return "zapier_runs"
        if "/templates/" in u:
            return "zapier_template"
        if "/editor" in u or "/app/" in u or "/draft" in u:
            return "zapier_editor"
        return "zapier"
    if "google.com/search" in u or "/search?q=" in u:
        return "search"
    if "accounts.google.com" in u:
        return "oauth"
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else "web"


_BROWSER_WINDOWS = ("chrome", "firefox", "edge", "safari", "browser")


def _is_browser(window: Optional[str]) -> bool:
    return bool(window) and any(b in window.lower() for b in _BROWSER_WINDOWS)


@dataclass
class Phase:
    key: str                 # group key (url-group for web, window for desktop)
    window: Optional[str]
    start: float
    end: float
    n_actions: int = 0
    urls: list[str] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    activity: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def typed_blob(self) -> str:
        return " \u23ce ".join(self.typed).lower()


def detect_phases(timeline: EventTimeline, min_seconds: float = 2.0) -> list[Phase]:
    """Group contiguous actions sharing the same app / URL-group into phases."""
    typed_by_t = {round(t["t"], 2): t["text"] for t in timeline.typed_texts(min_len=2)}

    phases: list[Phase] = []
    cur: Optional[Phase] = None
    last_url: Optional[str] = None

    for a in timeline.actions:
        if a.url:
            last_url = a.url
        if _is_browser(a.window):
            key = url_group(last_url) or "web"
        else:
            key = a.window or "(unknown)"

        if cur and cur.key == key:
            cur.end = a.t
            cur.n_actions += 1
        else:
            if cur:
                phases.append(cur)
            cur = Phase(key=key, window=a.window, start=a.t, end=a.t, n_actions=1)
        if a.url and a.url not in cur.urls:
            cur.urls.append(a.url)
        txt = typed_by_t.get(round(a.t, 2))
        if txt:
            cur.typed.append(txt)
    if cur:
        phases.append(cur)

    # Merge phases shorter than min_seconds into the previous phase to reduce
    # noise from quick app flickers, keeping the dominant key.
    merged: list[Phase] = []
    for p in phases:
        if merged and p.duration < min_seconds and p.key in ("search", "oauth", "web", "(unknown)"):
            prev = merged[-1]
            prev.end = p.end
            prev.n_actions += p.n_actions
            prev.urls.extend(u for u in p.urls if u not in prev.urls)
            prev.typed.extend(p.typed)
        else:
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# Phase -> activity classification
# ---------------------------------------------------------------------------
def classify_phase(phase: Phase, seen_zapier: bool) -> str:
    """Map a phase to a semantic activity used to build episodes."""
    g = phase.key
    blob = phase.typed_blob()

    if g in ("WPS Office", "Windows Explorer", "postwork"):
        return "source_review"
    if g == "google_sheets":
        return "validate" if seen_zapier else "create_sheet"
    if g == "slack":
        return "create_channel"
    if g == "zapier_publish":
        return "publish_zap"
    if g == "zapier_runs":
        return "validate"
    if g == "zapier_template":
        return "document"
    if g in ("zapier_editor", "zapier"):
        action_kw = ("slack", "send channel", "new booking alert", "message",
                     "group size", "boat assignment", "prepare the boat")
        if any(k in blob for k in action_kw):
            return "zap_action"
        return "zap_trigger"
    return "other"


# ---------------------------------------------------------------------------
# Episode assembly
# ---------------------------------------------------------------------------
# Per-activity templates: (episode id, primary app, grading checks, default
# reward weights, produced-artifact flag).
_ACTIVITY_SPEC = {
    "create_sheet": {
        "id": "E1_create_sheet", "app": "spreadsheet",
        "checks": ["working_sheet_populated"],
        "weights": {"working_sheet_populated": 1.0}, "produces": "sheet",
    },
    "create_channel": {
        "id": "E2_create_channel", "app": "slack",
        "checks": ["channel_exists"],
        "weights": {"channel_exists": 1.0}, "produces": "channel",
    },
    "zap_trigger": {
        "id": "E3_zap_trigger", "app": "zapier",
        "checks": ["trigger_configured"],
        "weights": {"trigger_configured": 1.0}, "produces": "trigger",
    },
    "zap_action": {
        "id": "E4_zap_action", "app": "zapier",
        "checks": ["action_configured"],
        "weights": {"action_configured": 1.0}, "produces": "action",
    },
    "publish_zap": {
        "id": "E5_publish_zap", "app": "zapier",
        "checks": ["zap_enabled"],
        "weights": {"zap_enabled": 1.0}, "produces": "published",
    },
    "validate": {
        "id": "E6_validate", "app": "spreadsheet",
        "checks": ["new_row_added", "alert_correct"],
        "weights": {"new_row_added": 0.4, "alert_correct": 0.6}, "produces": "",
    },
}

# The order episodes should appear in even if the recording revisits apps.
_EPISODE_ORDER = ["create_sheet", "create_channel", "zap_trigger",
                  "zap_action", "publish_zap", "validate"]


def _labels(activity: str, meta: dict) -> dict:
    cols = ", ".join(meta.get("columns", []))
    sheet = meta.get("sheet_title", "the booking sheet")
    target = meta.get("target_channel", "#crew-alerts")
    n = meta.get("n_rows", 0)
    table = {
        "create_sheet": {
            "title": "Create and populate the booking sheet",
            "goal": f"Create the '{sheet}' spreadsheet with columns ({cols}) and "
                    f"enter all {n} booking records from the source data.",
            "success_criteria": [
                f"A working sheet titled '{sheet}' exists.",
                f"It contains all {n} source booking rows with matching column values.",
            ],
        },
        "create_channel": {
            "title": "Create the Slack alerts channel",
            "goal": f"Create the {target} Slack channel that booking alerts post to.",
            "success_criteria": [f"A Slack channel named {target} exists in the workspace."],
        },
        "zap_trigger": {
            "title": "Configure the Zap trigger",
            "goal": f"Create a Zap whose trigger is Google Sheets 'New Spreadsheet Row' "
                    f"on the '{sheet}' sheet.",
            "success_criteria": [
                "The automation trigger is Google Sheets / New Spreadsheet Row on the booking sheet.",
            ],
        },
        "zap_action": {
            "title": "Configure the Zap Slack action",
            "goal": f"Add a Slack 'Send Channel Message' action that posts a formatted "
                    f"New Booking Alert to {target}.",
            "success_criteria": [
                f"The action targets {target}.",
                "The message template includes guest name, group size, experience type, "
                "boat assignment, start time, and a prepare-the-crew instruction.",
            ],
        },
        "publish_zap": {
            "title": "Publish and enable the Zap",
            "goal": "Turn the fully-configured Zap on so it runs automatically.",
            "success_criteria": ["The automation is enabled and fully configured."],
        },
        "validate": {
            "title": "Validate with a new booking",
            "goal": f"Add a new booking row and confirm a correctly formatted alert "
                    f"posts to {target}.",
            "success_criteria": [
                "A new booking row is added beyond the seeded rows.",
                f"A correct 'New Booking Alert' for that booking appears in {target}.",
            ],
        },
    }
    return table[activity]


def _overrides_for(flags: dict, meta: dict) -> dict:
    """Cumulative initial state at an episode's start, from produced flags."""
    channels = ["#general"]
    if flags.get("channel"):
        channels.append(meta.get("target_channel", "#crew-alerts"))
    if flags.get("published"):
        level = "full_enabled"
    elif flags.get("action"):
        level = "trigger_action"
    elif flags.get("trigger"):
        level = "trigger"
    else:
        level = "none"
    return {
        "working_sheet_rows": meta.get("n_rows", 0) if flags.get("sheet") else 0,
        "channels": channels,
        "automation_level": level,
    }


def build_episodes(phases: list[Phase], meta: dict) -> list[dict]:
    """Turn classified phases into ordered episodes with cumulative state."""
    # Classify phases, tracking when the first Zapier activity appears so a
    # Google Sheets phase after it counts as validation rather than setup.
    seen_zapier = False
    for p in phases:
        p.activity = classify_phase(p, seen_zapier)
        if p.key.startswith("zapier"):
            seen_zapier = True

    # Pick a representative phase per activity (first for setup activities, the
    # union of ranges for "validate").
    chosen: dict[str, dict] = {}
    for p in phases:
        act = p.activity
        if act not in _ACTIVITY_SPEC:
            continue
        if act not in chosen:
            chosen[act] = {"start": p.start, "end": p.end}
        else:
            chosen[act]["start"] = min(chosen[act]["start"], p.start)
            chosen[act]["end"] = max(chosen[act]["end"], p.end)

    episodes: list[dict] = []
    flags: dict = {}
    for act in _EPISODE_ORDER:
        if act not in chosen:
            continue
        spec = _ACTIVITY_SPEC[act]
        overrides = _overrides_for(flags, meta)
        labels = _labels(act, meta)
        episodes.append({
            "id": spec["id"],
            "activity": act,
            "app": spec["app"],
            "title": labels["title"],
            "goal": labels["goal"],
            "success_criteria": labels["success_criteria"],
            "time_range": [round(chosen[act]["start"], 1), round(chosen[act]["end"], 1)],
            "initial_overrides": overrides,
            "checks": spec["checks"],
            "reward_weights": spec["weights"],
            "produced_artifacts": [spec["produces"]] if spec["produces"] else [],
        })
        if spec["produces"]:
            flags[spec["produces"]] = True
    return episodes


# ---------------------------------------------------------------------------
# Canonical fallback (used if phase detection is too sparse)
# ---------------------------------------------------------------------------
def canonical_episodes(meta: dict) -> list[dict]:
    flags: dict = {}
    episodes: list[dict] = []
    for act in _EPISODE_ORDER:
        spec = _ACTIVITY_SPEC[act]
        labels = _labels(act, meta)
        episodes.append({
            "id": spec["id"], "activity": act, "app": spec["app"],
            "title": labels["title"], "goal": labels["goal"],
            "success_criteria": labels["success_criteria"],
            "time_range": None,
            "initial_overrides": _overrides_for(flags, meta),
            "checks": spec["checks"], "reward_weights": spec["weights"],
            "produced_artifacts": [spec["produces"]] if spec["produces"] else [],
        })
        if spec["produces"]:
            flags[spec["produces"]] = True
    return episodes


# ---------------------------------------------------------------------------
# Optional VLM label enrichment
# ---------------------------------------------------------------------------
def enrich_labels_llm(episodes: list[dict], evidence: dict,
                      model: str = "gpt-4o-mini") -> list[dict]:
    """Use an LLM to rewrite title/goal/success_criteria from the digest.

    Structure (ids, time ranges, overrides, checks) is preserved; only the
    human-readable fields may change. No-op if no credentials.
    """
    try:
        import vlm  # reuses .env loading + credential check
        if not vlm.have_credentials():
            return episodes
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        return episodes

    skeleton = [{"id": e["id"], "activity": e["activity"], "app": e["app"],
                 "time_range": e["time_range"]} for e in episodes]
    sys = ("You label RL sub-task episodes segmented from a screen recording. "
           "For each episode return a concise title, a one-sentence goal, and "
           "2-3 objectively checkable success_criteria. Keep ids unchanged. "
           "Output JSON {\"episodes\":[{\"id\",\"title\",\"goal\","
           "\"success_criteria\":[...]}]} only.")
    user = ("# OVERALL SESSION\n" + json.dumps(evidence.get("summary", {}), indent=2) +
            "\n\n# ACTION DIGEST\n" + evidence.get("action_digest", "")[:6000] +
            "\n\n# EPISODES TO LABEL\n" + json.dumps(skeleton, indent=2))
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}], max_tokens=1500)
        data = json.loads(resp.choices[0].message.content)
        by_id = {e["id"]: e for e in data.get("episodes", [])}
        for ep in episodes:
            patch = by_id.get(ep["id"])
            if patch:
                ep["title"] = patch.get("title", ep["title"])
                ep["goal"] = patch.get("goal", ep["goal"])
                if patch.get("success_criteria"):
                    ep["success_criteria"] = patch["success_criteria"]
    except Exception:
        pass
    return episodes


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def segment_timeline(timeline: EventTimeline, meta: dict,
                     evidence: Optional[dict] = None,
                     label_llm: bool = False) -> dict:
    """Produce the episode list (+ phase debug) for a timeline."""
    phases = detect_phases(timeline)
    episodes = build_episodes(phases, meta)
    if len(episodes) < 4:
        episodes = canonical_episodes(meta)
    if label_llm and evidence is not None:
        episodes = enrich_labels_llm(episodes, evidence)
    return {
        "episodes": episodes,
        "phases": [{
            "key": p.key, "activity": p.activity,
            "start": round(p.start, 1), "end": round(p.end, 1),
            "n_actions": p.n_actions,
        } for p in phases],
    }
