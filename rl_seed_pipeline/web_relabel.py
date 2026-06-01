"""Web-relabeling: rewrite the desktop parts of a trajectory as web actions.

The environment is intentionally **web-only** (the apps are simulated web apps
driven by Playwright). A recording, however, mixes a few desktop moments
(viewing the source spreadsheet in WPS Office, a Windows Explorer file open, and
the "postwork" brief app) with the rest, which is already in Chrome.

This module maps those non-Chrome segments onto web equivalents so the whole
trajectory reads as if it happened on websites:

* WPS Office (the source data)      -> a read-only web "source" spreadsheet,
* Windows Explorer (open the file)  -> navigate to that source URL,
* postwork (the task brief)         -> an in-app instructions panel.

Two products are produced:
1. ``relabel_report`` — a per-window summary of how desktop time was mapped.
2. an optional synthetic ``web_events.jsonl`` — the event stream rewritten so
   every action has ``active_window == "Google Chrome"`` and a web URL.

Relabeling is done at the **semantic** layer: pixel coordinates are kept but
flagged ``coords_approximate`` (a desktop app and a web app never share a
layout), because an RL seed needs the initial state + goal, not a pixel-faithful
action trace. Synthetic events are clearly marked so they are never confused
with captured ground truth.
"""

from __future__ import annotations

import json
from typing import Optional

from events import EventTimeline

# Logical web routes the env serves. Kept relative so they work behind any host.
SOURCE_ROUTE = "/source"        # read-only source dataset (was WPS Office)
SPREADSHEET_ROUTE = "/spreadsheet"
BRIEF_ROUTE = "/brief"          # instructions panel (was the postwork brief)

# Map a desktop window to its web-native target.
DESKTOP_TO_WEB = {
    "WPS Office": {
        "app": "spreadsheet", "route": SOURCE_ROUTE,
        "note": "Source booking data, viewed as a read-only web spreadsheet "
                "instead of the desktop WPS Office file.",
    },
    "Windows Explorer": {
        "app": "spreadsheet", "route": SOURCE_ROUTE,
        "note": "Opening the source file is relabeled to navigating to the "
                "read-only source spreadsheet URL.",
    },
    "postwork": {
        "app": "instructions", "route": BRIEF_ROUTE,
        "note": "The task brief is presented as an in-app instructions panel.",
    },
}

_BROWSER_WINDOWS = ("chrome", "firefox", "edge", "safari", "browser")


def _is_browser(window: Optional[str]) -> bool:
    return bool(window) and any(b in window.lower() for b in _BROWSER_WINDOWS)


def relabel_report(timeline: EventTimeline) -> dict:
    """Summarize how each desktop segment maps onto the web environment."""
    segments = timeline.app_segments()
    mapped = []
    desktop_seconds = 0.0
    for s in segments:
        if _is_browser(s.window):
            continue
        target = DESKTOP_TO_WEB.get(s.window)
        desktop_seconds += s.duration
        mapped.append({
            "window": s.window,
            "time_range": [round(s.start, 1), round(s.end, 1)],
            "duration": round(s.duration, 1),
            "n_actions": s.n_actions,
            "web_app": target["app"] if target else "spreadsheet",
            "web_route": target["route"] if target else SOURCE_ROUTE,
            "note": target["note"] if target else
                    "Unrecognized desktop app mapped to the source view by default.",
        })
    total = timeline.duration or 1.0
    return {
        "policy": "web-only; desktop segments relabeled to web at the semantic layer",
        "desktop_seconds": round(desktop_seconds, 1),
        "web_seconds": round(total - desktop_seconds, 1),
        "desktop_fraction": round(desktop_seconds / total, 4),
        "segments": mapped,
        "routes": {
            "source_data": SOURCE_ROUTE,
            "working_sheet": SPREADSHEET_ROUTE,
            "brief": BRIEF_ROUTE,
        },
    }


def relabel_events(timeline: EventTimeline) -> list[dict]:
    """Rewrite the raw events so the whole stream is web-native.

    Browser events are passed through (marked ``synthetic: False``). Desktop
    events get ``active_window = "Google Chrome"``, a web URL, and provenance
    markers. Coordinates are preserved but flagged approximate.
    """
    out: list[dict] = []
    for ev in timeline.raw_events:
        win = ev.get("active_window")
        new = dict(ev)
        if _is_browser(win):
            new["synthetic"] = False
            out.append(new)
            continue
        target = DESKTOP_TO_WEB.get(win, DESKTOP_TO_WEB["Windows Explorer"])
        new["active_window"] = "Google Chrome"
        new["active_window_url"] = target["route"]
        new["synthetic"] = True
        new["relabeled_from"] = win
        new["relabel_note"] = target["note"]
        new["web_app"] = target["app"]
        if any(k in ev for k in ("x", "y", "start_x", "end_x")):
            new["coords_approximate"] = True
        out.append(new)
    return out


def write_web_events(timeline: EventTimeline, path: str) -> int:
    """Write the synthetic web-native event stream as JSONL. Returns count."""
    events = relabel_events(timeline)
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return len(events)
