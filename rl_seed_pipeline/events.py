"""Deterministic event-stream analysis.

Parses a ``*_processed_events.jsonl`` file produced by the screen-recording
capture tool and turns the raw low-level events (mouse clicks, key presses,
typing, scrolls, drags) into structured, human/LLM-readable signals:

* a normalized, time-ordered list of actions (with video offsets in seconds),
* reconstructed typed text (handling Backspace / Shift / Space / Enter / Tab),
* application "segments" (contiguous runs in the same active window),
* the ordered list of URLs that were visited,
* a compact "action digest" that is small enough to drop into an LLM prompt,
* a set of candidate key-frame timestamps (the moments that best summarize the
  session) used later to pull screenshots for the vision model.

This stage uses no ML and no network -- it is fully deterministic and cheap,
so it is the backbone of the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Events whose ``key`` is a bare modifier; ignored when reconstructing text.
_MODIFIER_KEYS = {
    "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight",
    "AltLeft", "AltRight", "MetaLeft", "MetaRight", "Shift", "Control",
    "Alt", "Meta", "CapsLock",
}

# Punctuation produced when Shift is held with a digit/symbol key (US layout).
_SHIFT_SYMBOLS = {
    "1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^",
    "7": "&", "8": "*", "9": "(", "0": ")", "-": "_", "=": "+",
    "[": "{", "]": "}", "\\": "|", ";": ":", "'": '"', ",": "<",
    ".": ">", "/": "?", "`": "~",
}


def _is_nan(value: Any) -> bool:
    """The capture tool serializes missing URLs as the string ``"nan"``."""
    if value is None:
        return True
    if isinstance(value, float):
        return value != value  # NaN
    if isinstance(value, str):
        return value.strip().lower() in {"nan", "none", ""}
    return False


def _clean_url(value: Any) -> Optional[str]:
    return None if _is_nan(value) else str(value)


@dataclass
class Action:
    """A single normalized user action on the timeline."""

    index: int
    type: str
    t: float                       # best-estimate offset into the video (seconds)
    window: Optional[str] = None
    url: Optional[str] = None
    detail: str = ""               # short human-readable description
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "type": self.type,
            "t": round(self.t, 2),
            "window": self.window,
            "detail": self.detail,
        }
        if self.url:
            d["url"] = self.url
        return d


@dataclass
class AppSegment:
    """A contiguous span of time spent in a single application window."""

    window: str
    start: float
    end: float
    n_actions: int
    urls: list[str]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "n_actions": self.n_actions,
            "urls": self.urls,
        }


def _event_offset(ev: dict) -> Optional[float]:
    """Best-effort mapping of an event to a video offset in seconds."""
    for key in ("screenshot_at_start_offset", "screenshot_before_offset",
                "screenshot_after_offset"):
        v = ev.get(key)
        if isinstance(v, (int, float)) and not _is_nan(v):
            return float(v)
    return None


def reconstruct_typed_text(keys: Iterable[dict]) -> str:
    """Turn a ``type`` event's key list into the text the user most likely typed.

    Handles Backspace (delete previous char), Space, Enter (newline), Tab, and
    Shift (capitalization / shifted symbols). Other named keys (arrows, etc.)
    are dropped since they do not contribute characters.
    """
    out: list[str] = []
    for k in keys:
        key = k.get("key", "")
        mods = k.get("modifiers", []) or []
        shifted = "Shift" in mods
        ctrl = any(m in mods for m in ("Control", "Meta"))

        if key == "Backspace":
            if out:
                out.pop()
            continue
        if ctrl:
            # Ctrl/Cmd shortcuts (e.g. Ctrl+A, Ctrl+C) are commands, not text.
            continue
        if key == "Space":
            out.append(" ")
        elif key == "Enter":
            out.append("\n")
        elif key == "Tab":
            out.append("\t")
        elif len(key) == 1:
            if shifted:
                out.append(_SHIFT_SYMBOLS.get(key, key.upper()))
            else:
                out.append(key.lower())
        # else: named non-text key -> ignore
    return "".join(out)


def _describe(ev: dict, typed: str = "") -> str:
    """Short human-readable description for an event."""
    t = ev["type"]
    if t in ("mouse_click", "mouse_double_click", "mouse_triple_click"):
        kind = {"mouse_click": "click", "mouse_double_click": "double-click",
                "mouse_triple_click": "triple-click"}[t]
        return f"{kind} at ({int(ev.get('x', 0))},{int(ev.get('y', 0))})"
    if t == "mouse_drag":
        return (f"drag ({int(ev.get('start_x', 0))},{int(ev.get('start_y', 0))})"
                f"->({int(ev.get('end_x', 0))},{int(ev.get('end_y', 0))})")
    if t == "mouse_wheel":
        return "scroll"
    if t == "hold_key":
        return f"hold {ev.get('key', '')}"
    if t == "key_press":
        mods = ev.get("modifiers", []) or []
        combo = "+".join(list(mods) + [str(ev.get("key", ""))])
        return f"key {combo}"
    if t == "type":
        preview = typed.replace("\n", "\u23ce")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        return f'type "{preview}"'
    if t == "hover":
        return f"hover at ({int(ev.get('x', 0))},{int(ev.get('y', 0))})"
    return t


class EventTimeline:
    """Structured view over a processed-events JSONL file."""

    def __init__(self, events: list[dict]):
        self.raw_events = events
        self.actions: list[Action] = []
        self._build()

    # ---- construction -------------------------------------------------
    @classmethod
    def from_jsonl(cls, path: str) -> "EventTimeline":
        events = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return cls(events)

    def _build(self) -> None:
        last_t = 0.0
        for i, ev in enumerate(self.raw_events):
            t = _event_offset(ev)
            if t is None:
                t = last_t
            last_t = t
            typed = ""
            if ev.get("type") == "type":
                typed = reconstruct_typed_text(ev.get("keys", []))
            action = Action(
                index=i,
                type=ev.get("type", "unknown"),
                t=t,
                window=ev.get("active_window") or None,
                url=_clean_url(ev.get("active_window_url")),
                detail=_describe(ev, typed),
                raw=ev,
            )
            if typed:
                action.raw = ev
                action._typed = typed  # type: ignore[attr-defined]
            self.actions.append(action)
        self.actions.sort(key=lambda a: a.t)

    # ---- derived views ------------------------------------------------
    @property
    def duration(self) -> float:
        if not self.actions:
            return 0.0
        return max(a.t for a in self.actions)

    def app_segments(self, gap_merge: float = 0.0) -> list[AppSegment]:
        """Collapse consecutive actions in the same window into segments."""
        segments: list[AppSegment] = []
        cur: Optional[dict] = None
        for a in self.actions:
            win = a.window or "(unknown)"
            if cur and cur["window"] == win:
                cur["end"] = a.t
                cur["n"] += 1
                if a.url and a.url not in cur["urls"]:
                    cur["urls"].append(a.url)
            else:
                if cur:
                    segments.append(AppSegment(cur["window"], cur["start"],
                                               cur["end"], cur["n"], cur["urls"]))
                cur = {"window": win, "start": a.t, "end": a.t, "n": 1,
                       "urls": [a.url] if a.url else []}
        if cur:
            segments.append(AppSegment(cur["window"], cur["start"], cur["end"],
                                       cur["n"], cur["urls"]))
        return segments

    def typed_texts(self, min_len: int = 1) -> list[dict]:
        """All reconstructed typed strings, in order, with timing + context."""
        result = []
        for a in self.actions:
            if a.type == "type":
                txt = getattr(a, "_typed", "")
                if len(txt.strip()) >= min_len:
                    result.append({
                        "t": round(a.t, 2),
                        "window": a.window,
                        "url": a.url,
                        "text": txt,
                    })
        return result

    def visited_urls(self) -> list[dict]:
        """Ordered, de-duplicated list of URLs with first-seen offset."""
        seen: dict[str, float] = {}
        for a in self.actions:
            if a.url and a.url not in seen:
                seen[a.url] = a.t
        return [{"t": round(t, 2), "url": u}
                for u, t in sorted(seen.items(), key=lambda kv: kv[1])]

    def domains(self) -> list[str]:
        doms = []
        for a in self.actions:
            if a.url:
                m = re.match(r"https?://([^/]+)", a.url)
                if m and m.group(1) not in doms:
                    doms.append(m.group(1))
        return doms

    # ---- key-frame selection -----------------------------------------
    def keyframe_offsets(self, max_frames: int = 12) -> list[float]:
        """Pick the most informative video offsets to screenshot.

        Strategy: always include the very first frame (initial state), the
        first action inside each new application segment (context switches),
        the first action after each URL *domain* change (web sub-phases such as
        Sheets -> Slack -> Zapier -> OAuth), and a few evenly spaced samples to
        cover long uninterrupted stretches.
        """
        dur = self.duration

        # MANDATORY frames: the initial state + the first frame of every app
        # segment (context switches). These are few but high-signal -- e.g. a
        # short visit to a source document (WPS Office) where the seed data is
        # fully visible must never be dropped.
        mandatory: set[float] = set()
        if self.actions:
            mandatory.add(round(self.actions[0].t, 2))
        for seg in self.app_segments():
            mandatory.add(round(seg.start, 2))

        # OPTIONAL frames: web sub-phase changes (domain transitions) plus
        # evenly spaced time samples to cover long uninterrupted stretches.
        optional: set[float] = set()
        prev_dom = None
        for a in self.actions:
            if a.url:
                m = re.match(r"https?://([^/]+)", a.url)
                dom = m.group(1) if m else None
                if dom and dom != prev_dom:
                    optional.add(round(a.t, 2))
                    prev_dom = dom
        if dur > 0:
            for k in range(max_frames + 1):
                optional.add(round(dur * k / max_frames, 2))
        optional -= mandatory

        # If mandatory alone already exceeds the budget, thin it time-uniformly
        # but always keep the very first frame.
        if len(mandatory) >= max_frames:
            ms = sorted(mandatory)
            keep = {ms[0]}
            for k in range(max_frames):
                target = dur * k / max(1, max_frames - 1)
                keep.add(min(ms, key=lambda o: abs(o - target)))
            return sorted(keep)

        # Otherwise keep ALL mandatory frames and fill the remaining budget with
        # time-uniformly spread optional frames.
        result = set(mandatory)
        remaining = max_frames - len(result)
        opt = sorted(optional)
        for k in range(remaining):
            if not opt:
                break
            target = dur * k / max(1, remaining - 1) if dur > 0 else 0
            cand = min(opt, key=lambda o: abs(o - target))
            result.add(cand)
            opt.remove(cand)
        return sorted(result)

    def screenshot_urls(self) -> list[dict]:
        """Collect remote screenshot URLs (Azure) keyed by offset.

        These are higher quality than decoded video frames and are already
        aligned to meaningful moments (before/after each action).
        """
        out = []
        for a in self.actions:
            ev = a.raw
            for field_name, off_field in (
                ("screenshot_at_start_url", "screenshot_at_start_offset"),
                ("screenshot_before_url", "screenshot_before_offset"),
                ("screenshot_after_url", "screenshot_after_offset"),
            ):
                url = _clean_url(ev.get(field_name))
                off = ev.get(off_field)
                if url and isinstance(off, (int, float)):
                    out.append({
                        "t": round(float(off), 2),
                        "url": url,
                        "window": a.window,
                        "action_index": a.index,
                        "kind": field_name.replace("screenshot_", "").replace("_url", ""),
                    })
        out.sort(key=lambda d: d["t"])
        return out

    # ---- compact digest for LLM --------------------------------------
    def action_digest(self, max_lines: int = 120) -> str:
        """A compact, token-efficient transcript of the session.

        Consecutive scrolls/holds are collapsed; window changes are annotated.
        """
        lines: list[str] = []
        prev_win = None
        pending_scroll = 0
        for a in self.actions:
            if a.type == "mouse_wheel":
                pending_scroll += 1
                continue
            if a.type == "hold_key":
                # holds are usually part of a shortcut already captured by key/type
                continue
            if pending_scroll:
                lines.append(f"      ... scrolled x{pending_scroll}")
                pending_scroll = 0
            if a.window != prev_win:
                tag = f"\n=== {a.window or '(unknown)'} ==="
                if a.url:
                    tag += f"  [{a.url.split('?')[0]}]"
                lines.append(tag)
                prev_win = a.window
            stamp = f"{int(a.t // 60):02d}:{int(a.t % 60):02d}"
            line = f"  {stamp}  {a.detail}"
            if a.type in ("mouse_click", "mouse_double_click") and a.url:
                base = a.url.split("?")[0]
                if base not in (lines[-1] if lines else ""):
                    line += f"   [{base}]"
            lines.append(line)
        if pending_scroll:
            lines.append(f"      ... scrolled x{pending_scroll}")

        if len(lines) > max_lines:
            head = lines[: max_lines // 2]
            tail = lines[-max_lines // 2:]
            lines = head + [f"\n  ... ({len(lines) - max_lines} lines omitted) ...\n"] + tail
        return "\n".join(lines)

    def summary(self) -> dict:
        segs = self.app_segments()
        by_app: dict[str, float] = {}
        for s in segs:
            by_app[s.window] = by_app.get(s.window, 0.0) + s.duration
        return {
            "duration_seconds": round(self.duration, 2),
            "n_events": len(self.actions),
            "applications": sorted(by_app, key=by_app.get, reverse=True),
            "time_per_application": {k: round(v, 1) for k, v in
                                     sorted(by_app.items(), key=lambda kv: kv[1],
                                            reverse=True)},
            "domains": self.domains(),
            "n_app_switches": len(segs),
        }
