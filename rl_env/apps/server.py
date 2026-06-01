"""Standalone HTTP server hosting the three simulated apps + a small REST API.

Run directly:
    python3 apps/server.py --seed ../seed.example.json --port 8765

State is held in-memory and rebuilt from the seed on POST /api/reset, which makes
every RL episode deterministic and the whole environment resettable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import render  # noqa: E402
from state import SimState, load_seed  # noqa: E402

ORIGINAL_SEED: dict = {}
EPISODE_ID = None  # type: ignore
STATE: SimState = None  # type: ignore
STATE_LOCK = threading.RLock()


def rebuild(seed: dict) -> None:
    global STATE
    with STATE_LOCK:
        STATE = SimState.from_seed(seed, episode_id=EPISODE_ID)


class Handler(BaseHTTPRequestHandler):
    server_version = "SaltStoneSim/1.0"

    # ----- helpers ------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _html(self, html_text: str, code: int = 200) -> None:
        self._send(code, html_text.encode("utf-8"))

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_body(self) -> tuple[dict, bool]:
        """Return (params, is_form). Supports form-encoded and JSON bodies."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        ctype = (self.headers.get("Content-Type", "") or "").lower()
        if "application/json" in ctype:
            try:
                return (json.loads(raw.decode("utf-8") or "{}"), False)
            except json.JSONDecodeError:
                return ({}, False)
        parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        flat = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        return (flat, True)

    def log_message(self, fmt, *args):  # noqa: A003 - silence default logging
        if os.environ.get("SIM_VERBOSE"):
            super().log_message(fmt, *args)

    # ----- routing ------------------------------------------------------
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/healthz", "/health"):
            return self._json({"ok": True})
        if path == "/":
            return self._redirect("/spreadsheet")
        if path == "/spreadsheet":
            return self._html(render.render_spreadsheet(STATE))
        if path == "/source":
            return self._html(render.render_source(STATE))
        if path == "/brief":
            return self._html(render.render_brief(STATE))
        if path == "/slack":
            qs = parse_qs(parsed.query)
            channel = qs.get("channel", [None])[0]
            return self._html(render.render_slack(STATE, channel))
        if path == "/zapier":
            return self._html(render.render_zapier(STATE))
        if path == "/api/state":
            return self._json(STATE.to_dict())
        if path == "/api/seed":
            return self._json(ORIGINAL_SEED)
        return self._json({"error": "not found", "path": path}, code=404)

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        params, is_form = self._read_body()

        if path == "/api/reset":
            seed = params.get("seed") if isinstance(params, dict) else None
            rebuild(seed if isinstance(seed, dict) and seed else ORIGINAL_SEED)
            if is_form:
                return self._redirect("/spreadsheet")
            return self._json({"ok": True, "state": STATE.to_dict()})

        if path == "/api/spreadsheet/rows":
            values = params.get("values") if isinstance(params.get("values"), dict) else params
            result = STATE.add_row(values)
            if is_form:
                return self._redirect("/spreadsheet")
            return self._json({"ok": True, **result})

        if path == "/api/automation":
            if params.get("load_recommended"):
                # Fill the template field only; keep other typed values, don't enable.
                patch = {
                    "channel": params.get("channel", STATE.automation.channel),
                    "bot_name": params.get("bot_name", STATE.automation.bot_name),
                    "message_template": STATE.recommended_template(),
                    "enabled": False,
                }
                STATE.update_automation(patch)
                if is_form:
                    return self._redirect("/zapier")
                return self._json({"ok": True, "automation": STATE.automation.to_dict()})
            patch = dict(params)
            if is_form and "enabled" not in params:
                patch["enabled"] = False  # unchecked checkboxes are omitted by browsers
            result = STATE.update_automation(patch)
            if is_form:
                return self._redirect("/zapier")
            return self._json({"ok": True, "automation": result})

        if path == "/api/slack/channels":
            name = params.get("name") or params.get("channel") or ""
            result = STATE.create_channel(name)
            if is_form:
                ch = (result.get("channel") or "").lstrip("#")
                return self._redirect(f"/slack?channel={ch}" if ch else "/slack")
            return self._json({"ok": True, **result})

        if path == "/api/slack/messages":
            msg = STATE.post_message(
                params.get("channel", STATE.target_channel),
                params.get("text", ""),
                params.get("author", "you"),
            )
            if is_form:
                return self._redirect("/slack")
            return self._json({"ok": True, "message": msg})

        return self._json({"error": "not found", "path": path}, code=404)


def main() -> None:
    ap = argparse.ArgumentParser(description="Salt & Stone simulated apps server")
    ap.add_argument("--seed", required=True, help="path to seed JSON")
    ap.add_argument("--episode", default=None,
                    help="episode id to reset to (default: full task)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    global ORIGINAL_SEED, EPISODE_ID
    ORIGINAL_SEED = load_seed(args.seed)
    EPISODE_ID = args.episode
    rebuild(ORIGINAL_SEED)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f"[sim] serving on http://{args.host}:{args.port} (seed={args.seed})\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
