"""Media acquisition: key-frame screenshots for the vision model.

Two sources are supported, in priority order:

1. **Remote screenshots** captured by the recording tool (``screenshot_*_url``
   fields in the events). These are already aligned to meaningful moments
   (before/after each action) and are the cheapest, highest-signal frames.
2. **Decoded video frames** pulled from the ``.webm`` with ffmpeg at chosen
   offsets. Used as a fallback when remote screenshots are unavailable.

The module exposes :func:`select_keyframes`, which combines the timeline's
key-frame offsets with application-segment labels so each frame handed to the
vision model carries a short caption (offset + active app + nearby action).
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Optional

from events import EventTimeline


@dataclass
class KeyFrame:
    offset: float
    label: str           # e.g. "WPS Office" / "Google Chrome (docs.google.com)"
    note: str            # nearest action detail / phase hint
    source: str          # "remote" | "video"
    url: Optional[str] = None
    path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "offset": round(self.offset, 2),
            "label": self.label,
            "note": self.note,
            "source": self.source,
            "url": self.url,
            "path": self.path,
        }


def _nearest_remote_shot(shots: list[dict], offset: float,
                         tol: float = 6.0) -> Optional[dict]:
    if not shots:
        return None
    best = min(shots, key=lambda s: abs(s["t"] - offset))
    return best if abs(best["t"] - offset) <= tol else None


def select_keyframes(timeline: EventTimeline, max_frames: int = 12) -> list[KeyFrame]:
    """Choose informative frames and label them with app + nearby action."""
    import re as _re
    offsets = timeline.keyframe_offsets(max_frames=max_frames)
    shots = timeline.screenshot_urls()

    def _nearest_action(t: float):
        return min(timeline.actions, key=lambda a: abs(a.t - t), default=None)

    def seg_for(t: float) -> str:
        """Label = active window of the nearest action + (for browsers) the
        most recent URL domain at/just-before this offset."""
        near = _nearest_action(t)
        if not near:
            return "(unknown)"
        win = near.window or "(unknown)"
        is_browser = any(b in win.lower() for b in ("chrome", "firefox",
                         "edge", "safari", "browser"))
        if not is_browser:
            return win
        dom = None
        for a in timeline.actions:
            if a.t <= t + 1 and a.url:
                m = _re.match(r"https?://([^/]+)", a.url)
                if m:
                    dom = m.group(1)
        return f"{win} ({dom})" if dom else win

    def note_for(t: float) -> str:
        near = _nearest_action(t)
        if near and abs(near.t - t) <= 8:
            return near.detail
        return ""

    frames: list[KeyFrame] = []
    for off in offsets:
        remote = _nearest_remote_shot(shots, off)
        if remote:
            frames.append(KeyFrame(offset=remote["t"], label=seg_for(remote["t"]),
                                   note=note_for(remote["t"]), source="remote",
                                   url=remote["url"]))
        else:
            frames.append(KeyFrame(offset=off, label=seg_for(off),
                                   note=note_for(off), source="video"))
    # de-duplicate by rounded offset
    seen = set()
    uniq = []
    for f in frames:
        key = round(f.offset, 1)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def fetch_remote(frame: KeyFrame, out_dir: str, name: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    if not frame.url:
        return None
    path = os.path.join(out_dir, name)
    try:
        urllib.request.urlretrieve(frame.url, path)
        frame.path = path
        return path
    except Exception:
        return None


def extract_video_frame(video_path: str, offset: float, out_dir: str,
                        name: str) -> Optional[str]:
    """Pull a single frame at ``offset`` seconds using ffmpeg."""
    if not shutil.which("ffmpeg"):
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(offset),
           "-i", video_path, "-frames:v", "1", "-q:v", "3", path]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return path if os.path.exists(path) else None
    except Exception:
        return None


def materialize(frames: list[KeyFrame], out_dir: str,
                video_path: Optional[str] = None) -> list[KeyFrame]:
    """Download/extract every key-frame to ``out_dir`` and set ``.path``."""
    for i, f in enumerate(frames):
        name = f"kf_{i:02d}_{int(f.offset)}s.jpg"
        if f.source == "remote":
            if not fetch_remote(f, out_dir, name) and video_path:
                p = extract_video_frame(video_path, f.offset, out_dir, name)
                if p:
                    f.source, f.path = "video", p
        elif video_path:
            p = extract_video_frame(video_path, f.offset, out_dir, name)
            if p:
                f.path = p
    return frames


def encode_image_b64(path: str, max_side: int = 1280) -> Optional[str]:
    """Base64-encode an image (optionally downscaled) for a vision API call."""
    try:
        from PIL import Image
        import io
        im = Image.open(path).convert("RGB")
        if max(im.size) > max_side:
            ratio = max_side / max(im.size)
            im = im.resize((int(im.width * ratio), int(im.height * ratio)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            with open(path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            return None
