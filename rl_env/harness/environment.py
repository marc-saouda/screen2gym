"""Playwright-driven, resettable RL environment for the booking-alert task.

Boots the local simulated apps (spreadsheet / Slack / Zapier) from a seed JSON
and exposes a small RL API:

    env = SaltStoneEnv(seed_path="seed.example.json")
    obs = env.reset()
    obs = env.step({"type": "configure_automation", ...})
    obs = env.step({"type": "add_row", "values": {...}})
    env.reward()       -> float (shaped)
    env.is_success()   -> bool
    env.close()

The environment prefers a real Chromium browser via Playwright. If Playwright or
Chromium is unavailable it transparently degrades to a "backend" mode that drives
the same server over HTTP and snapshots rendered HTML instead of pixels, so the
full task logic still runs and can be graded.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

try:  # work both as a package and as a loose script
    from . import actions as actions_mod
    from . import judge as judge_mod
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import actions as actions_mod  # type: ignore
    import judge as judge_mod  # type: ignore

HARNESS_DIR = Path(__file__).resolve().parent
RL_ENV_DIR = HARNESS_DIR.parent
SERVER_PATH = RL_ENV_DIR / "apps" / "server.py"
DEFAULT_SEED = RL_ENV_DIR / "seed.example.json"
DEFAULT_OUTPUTS = RL_ENV_DIR / "outputs"

# Episode specs live with the apps (shared by the server and this grader).
sys.path.insert(0, str(RL_ENV_DIR / "apps"))
import episodes as episodes_mod  # noqa: E402

# Maps semantic column roles to the seed's column names so grading/auto-fill is
# robust to column reordering.
COLUMN_ROLES = {
    "id": "Booking_ID",
    "guest_name": "Guest_Name",
    "experience": "Experience_Type",
    "boat": "Boat_Name",
    "guest_count": "Guest_Count",
    "start_time": "Start_Time",
}


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SaltStoneEnv:
    def __init__(
        self,
        seed_path: str | os.PathLike = DEFAULT_SEED,
        mode: str = "auto",  # "auto" | "browser" | "backend"
        headless: bool = True,
        host: str = "127.0.0.1",
        port: int = 0,
        output_dir: str | os.PathLike = DEFAULT_OUTPUTS,
        use_llm_judge: bool = True,
        judge_model: str = "gpt-4o-mini",
        episode_id: Optional[str] = None,
    ) -> None:
        self.seed_path = str(Path(seed_path).resolve())
        self.requested_mode = mode
        self.mode = "backend"
        self.headless = headless
        self.host = host
        self.port = port or _free_port()
        self.base_url = f"http://{self.host}:{self.port}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model

        self.proc: Optional[subprocess.Popen] = None
        self._server_log = None
        self._pw = None
        self.browser = None
        self.context = None
        self.page = None
        self.browser_error: Optional[str] = None

        self.current_url = f"{self.base_url}/spreadsheet"
        self.step_index = 0
        self.last_eval: dict[str, Any] = {}
        self._judge_cache: dict[str, dict[str, Any]] = {}
        self._auto_row_counter = 0

        with open(self.seed_path, "r", encoding="utf-8") as fh:
            self.seed = json.load(fh)
        ds = self.seed.get("initial_state", {}).get("datasets", [{}])[0]
        self.columns: list[str] = list(ds.get("columns", []))
        self.target_channel = self.seed.get("initial_state", {}).get(
            "target_slack_channel", "#crew-alerts"
        )

        # Resolve the episode (None -> full task) and its grader spec.
        self.episode_id = episode_id
        self.episode = episodes_mod.find_episode(self.seed, episode_id)
        self.checks: list[str] = list(self.episode.get("checks", []))
        self.reward_weights: dict[str, float] = dict(self.episode.get("reward_weights", {}))
        self.success_checks: list[str] = list(
            self.episode.get("success_checks") or self.checks)

    # ================= lifecycle =================
    def _start_server(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self._server_log = open(self.output_dir / "server.log", "w")
        cmd = [
            sys.executable,
            str(SERVER_PATH),
            "--seed",
            self.seed_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.episode_id:
            cmd += ["--episode", self.episode_id]
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._server_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("simulated apps server exited during startup")
            try:
                r = requests.get(f"{self.base_url}/healthz", timeout=0.5)
                if r.ok:
                    return
            except requests.RequestException:
                time.sleep(0.15)
        raise RuntimeError("server did not become healthy in time")

    def _maybe_start_browser(self) -> None:
        if self.requested_mode == "backend":
            self.mode = "backend"
            self.browser_error = "backend mode requested"
            return
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            self.browser_error = f"playwright import failed: {exc}"
            self.mode = "backend"
            return
        try:
            self._pw = sync_playwright().start()
            self.browser = self._pw.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context(viewport={"width": 1280, "height": 920})
            self.page = self.context.new_page()
            self.page.set_default_timeout(8000)
            self.mode = "browser"
        except Exception as exc:
            self.browser_error = f"chromium launch failed: {exc}"
            self._teardown_browser()
            self.mode = "backend"

    def reset(self) -> dict[str, Any]:
        self._start_server()
        if self.page is None and self.browser_error is None:
            self._maybe_start_browser()
        # Reset backend state to the seed for a clean episode.
        requests.post(f"{self.base_url}/api/reset", json={}, timeout=5)
        self.step_index = 0
        self._judge_cache.clear()
        self._auto_row_counter = 0
        self.current_url = self._app_url(self.episode.get("app", "spreadsheet"))
        if self.mode == "browser":
            self.page.goto(self.current_url, wait_until="load")
        return self.observe(label="reset")

    def close(self) -> None:
        self._teardown_browser()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._server_log:
            try:
                self._server_log.close()
            except Exception:
                pass

    def _teardown_browser(self) -> None:
        for obj, meth in (
            (self.context, "close"),
            (self.browser, "close"),
            (self._pw, "stop"),
        ):
            try:
                if obj is not None:
                    getattr(obj, meth)()
            except Exception:
                pass
        self.context = self.browser = self._pw = self.page = None

    def __enter__(self) -> "SaltStoneEnv":
        self.reset()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ================= actions =================
    def _app_url(self, app: str, channel: Optional[str] = None) -> str:
        if app == "spreadsheet":
            return f"{self.base_url}/spreadsheet"
        if app == "zapier":
            return f"{self.base_url}/zapier"
        if app == "slack":
            ch = (channel or self.target_channel or "").lstrip("#")
            return f"{self.base_url}/slack" + (f"?channel={ch}" if ch else "")
        return f"{self.base_url}/{app}"

    def _submit(self, selector: str) -> None:
        """Click a submit control and wait for the resulting navigation."""
        try:
            with self.page.expect_navigation(wait_until="load"):
                self.page.click(selector)
        except Exception:
            self.page.wait_for_load_state("load")

    def _auto_row_values(self) -> dict[str, str]:
        self._auto_row_counter += 1
        n = self._auto_row_counter
        state = self._get_state()
        rows = state.get("dataset", {}).get("rows", [])
        best = 0
        for r in rows:
            v = str(r.get(COLUMN_ROLES["id"], ""))
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                best = max(best, int(digits))
        return {
            COLUMN_ROLES["id"]: f"SST-{best + 1:05d}",
            COLUMN_ROLES["guest_name"]: f"Test Guest {n}",
            COLUMN_ROLES["experience"]: "Private Fishing Charter",
            COLUMN_ROLES["boat"]: "Blue Marlin",
            COLUMN_ROLES["guest_count"]: str(3 + n),
            COLUMN_ROLES["start_time"]: f"2024-08-0{(n % 9) + 1}T07:30:00",
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        actions_mod.validate(action)
        atype = action["type"]
        self.step_index += 1
        info: dict[str, Any] = {"applied": atype}

        if atype == "navigate":
            url = self._app_url(action.get("app", "spreadsheet"), action.get("channel"))
            self.current_url = url
            if self.mode == "browser":
                self.page.goto(url, wait_until="load")

        elif atype == "add_row":
            values = action.get("values") or self._auto_row_values()
            values = {k: str(v) for k, v in values.items()}
            if self.mode == "browser":
                self.page.goto(self._app_url("spreadsheet"), wait_until="load")
                for col in self.columns:
                    sel = f"#cell-{col}"
                    try:
                        self.page.fill(sel, values.get(col, ""))
                    except Exception:
                        pass
                self._submit("#add-row-btn")
                self.current_url = self._app_url("spreadsheet")
            else:
                requests.post(
                    f"{self.base_url}/api/spreadsheet/rows",
                    json={"values": values},
                    timeout=5,
                )
            info["values"] = values

        elif atype == "create_channel":
            name = action.get("name") or self.target_channel
            if self.mode == "browser":
                self.page.goto(self._app_url("slack"), wait_until="load")
                try:
                    self.page.fill("#create-channel-input", str(name).lstrip("#"))
                    self._submit("#create-channel-btn")
                except Exception:
                    pass
                self.current_url = self._app_url("slack", name)
            else:
                requests.post(
                    f"{self.base_url}/api/slack/channels",
                    json={"name": name}, timeout=5,
                )
            info["channel"] = name

        elif atype == "configure_automation":
            self._configure_automation(action)
            info["channel"] = action.get("channel")

        elif atype == "click":
            if self.mode == "browser":
                self.page.click(action["selector"])
            else:
                info["warning"] = "click is browser-only; ignored in backend mode"

        elif atype == "fill":
            if self.mode == "browser":
                self.page.fill(action["selector"], action.get("text", ""))
            else:
                info["warning"] = "fill is browser-only; ignored in backend mode"

        elif atype in ("screenshot", "noop"):
            pass

        return self.observe(label=atype, info=info)

    def _configure_automation(self, action: dict[str, Any]) -> None:
        # Only fields explicitly present are applied, so the trigger and the
        # action can be configured independently across episodes.
        channel = action.get("channel")            # None => leave as-is
        template = action.get("message_template")  # None => leave as-is
        bot_name = action.get("bot_name", "Zapier")
        enabled = bool(action.get("enabled", True))
        use_recommended = bool(action.get("use_recommended_template", False))
        trig_app = action.get("trigger_app")
        trig_event = action.get("trigger_event")
        trig_sheet = action.get("trigger_sheet")

        if self.mode == "browser":
            self.page.goto(self._app_url("zapier"), wait_until="load")
            if use_recommended:
                self._submit("#zap-load-template")
            for sel, val in (("#zap-trigger-app", trig_app),
                             ("#zap-trigger-event", trig_event)):
                if val is not None:
                    try:
                        self.page.select_option(sel, val)
                    except Exception:
                        pass
            if trig_sheet is not None:
                try:
                    self.page.fill("#zap-trigger-sheet", trig_sheet)
                except Exception:
                    pass
            if channel is not None:
                self.page.fill("#zap-channel", channel or "")
            self.page.fill("#zap-bot", bot_name or "")
            if template is not None:
                self.page.fill("#zap-template", template)
            try:
                if enabled:
                    self.page.check("#zap-enabled")
                else:
                    self.page.uncheck("#zap-enabled")
            except Exception:
                pass
            self._submit("#zap-save")
            self.current_url = self._app_url("zapier")
            return

        # backend mode
        if use_recommended and template is None:
            rec = {"load_recommended": "1", "bot_name": bot_name}
            if channel is not None:
                rec["channel"] = channel
            requests.post(f"{self.base_url}/api/automation", json=rec, timeout=5)
        payload: dict[str, Any] = {"bot_name": bot_name, "enabled": enabled}
        if channel is not None:
            payload["channel"] = channel
        if template is not None:
            payload["message_template"] = template
        for key, val in (("trigger_app", trig_app), ("trigger_event", trig_event),
                         ("trigger_sheet", trig_sheet)):
            if val is not None:
                payload[key] = val
        requests.post(f"{self.base_url}/api/automation", json=payload, timeout=5)

    # ================= observation & grading =================
    def _get_state(self) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.base_url}/api/state", timeout=5)
            return r.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def observe(self, label: str = "", info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        state = self._get_state()
        evaluation = self._evaluate(state)
        self.last_eval = evaluation

        screenshot_path: Optional[str] = None
        dom_path: Optional[str] = None
        aria: Optional[Any] = None
        html_excerpt: Optional[str] = None

        prefix = f"{self.step_index:02d}_{label or 'obs'}"
        if self.mode == "browser" and self.page is not None:
            screenshot_path = str(self.output_dir / f"{prefix}.png")
            try:
                self.page.screenshot(path=screenshot_path, full_page=True)
            except Exception as exc:
                info = dict(info or {})
                info["screenshot_error"] = str(exc)
                screenshot_path = None
            try:
                aria = self.page.accessibility.snapshot()
            except Exception:
                aria = None
            self.current_url = self.page.url
        else:
            try:
                r = requests.get(self.current_url, timeout=5)
                html = r.text
                dom_path = str(self.output_dir / f"{prefix}.html")
                with open(dom_path, "w", encoding="utf-8") as fh:
                    fh.write(html)
                html_excerpt = _text_excerpt(html)
            except requests.RequestException as exc:
                html_excerpt = f"<fetch error: {exc}>"

        return {
            "step": self.step_index,
            "mode": self.mode,
            "url": self.current_url,
            "screenshot_path": screenshot_path,
            "dom_snapshot_path": dom_path,
            "aria": aria,
            "html_excerpt": html_excerpt,
            "state": state,
            "reward": evaluation["reward"],
            "success": evaluation["success"],
            "reward_breakdown": evaluation["breakdown"],
            "task": state.get("task", {}),
            "info": info or {},
        }

    @staticmethod
    def _norm(text: str) -> str:
        import re
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _sheet_populated(self, work_rows: list, src_rows: list,
                         columns: list[str]) -> bool:
        """Every source booking is present in the working sheet (by id + values)."""
        if not src_rows:
            return False
        id_col = COLUMN_ROLES.get("id") or (columns[0] if columns else "")
        by_id = {self._norm(str(r.get(id_col, ""))): r for r in work_rows}
        for s in src_rows:
            w = by_id.get(self._norm(str(s.get(id_col, ""))))
            if not w:
                return False
            for c in columns:
                if self._norm(str(w.get(c, ""))) != self._norm(str(s.get(c, ""))):
                    return False
        return True

    def _compute_checks(self, state: dict[str, Any]) -> tuple[dict[str, bool], dict, dict]:
        """Evaluate every grader check; episodes select the subset they need."""
        ds = state.get("dataset", {})
        src = state.get("source", {})
        auto = state.get("automation", {})
        slack = state.get("slack", {})
        columns = ds.get("columns", self.columns)
        rows = ds.get("rows", [])
        new_row_count = ds.get("new_row_count", 0)
        target = slack.get("target_channel", self.target_channel)
        channels = slack.get("channels", [])
        messages = slack.get("messages", [])
        chan_ok = auto.get("channel_normalized") == target

        checks: dict[str, bool] = {
            "working_sheet_populated": self._sheet_populated(rows, src.get("rows", []), columns),
            "channel_exists": target in channels,
            "trigger_configured": bool(auto.get("trigger_is_configured")),
            "action_configured": bool(auto.get("action_is_configured")) and chan_ok,
            "zap_enabled": bool(auto.get("enabled")) and bool(auto.get("configured")) and chan_ok,
            "new_row_added": new_row_count >= 1,
        }

        expected_row = rows[-1] if rows else {}
        alerts = [
            m for m in messages
            if m.get("channel") == target and (m.get("source") == "automation" or m.get("is_bot"))
        ]
        judge_detail: dict[str, Any] = {"note": "no alert in target channel"}
        c_alert = False
        if alerts and expected_row:
            chosen = self._pick_alert(alerts, expected_row, columns)
            judge_detail = self._judge_cached(chosen.get("text", ""), expected_row, columns)
            c_alert = bool(judge_detail.get("correct"))
        checks["alert_correct"] = c_alert

        extra = {
            "new_row_count": new_row_count,
            "target_channel": target,
            "channels": channels,
            "automation_channel": auto.get("channel_normalized"),
            "automation_enabled": auto.get("enabled"),
        }
        return checks, judge_detail, extra

    def _evaluate(self, state: dict[str, Any]) -> dict[str, Any]:
        if "error" in state:
            return {"reward": 0.0, "success": False, "breakdown": {"error": state["error"]}}

        checks, judge_detail, extra = self._compute_checks(state)
        names = self.checks or list(checks.keys())
        weights = self.reward_weights or {n: 1.0 for n in names}
        total_w = sum(weights.get(n, 0.0) for n in names) or 1.0
        got = sum(weights.get(n, 0.0) for n in names if checks.get(n))
        reward = round(got / total_w, 3)
        success = bool(self.success_checks) and all(
            checks.get(n) for n in self.success_checks)

        return {
            "reward": reward,
            "success": success,
            "breakdown": {
                "episode": self.episode_id or "full_task",
                "components": {n: checks.get(n, False) for n in names},
                "all_checks": checks,
                "weights": weights,
                "success_checks": self.success_checks,
                "judge": judge_detail,
                **extra,
            },
        }

    def _pick_alert(
        self, alerts: list[dict[str, Any]], expected_row: dict[str, str], columns: list[str]
    ) -> dict[str, Any]:
        name_col = COLUMN_ROLES.get("guest_name", columns[1] if len(columns) > 1 else "")
        guest = (expected_row.get(name_col, "") or "").lower()
        for m in reversed(alerts):
            if guest and guest in (m.get("text", "") or "").lower():
                return m
        return alerts[-1]

    def _judge_cached(
        self, text: str, row: dict[str, str], columns: list[str]
    ) -> dict[str, Any]:
        key = text
        if key in self._judge_cache:
            return self._judge_cache[key]
        detail = judge_mod.judge_alert(
            text,
            row,
            columns,
            use_llm=self.use_llm_judge,
            model=self.judge_model,
            column_roles=COLUMN_ROLES,
        )
        self._judge_cache[key] = detail
        return detail

    # ================= public grading API =================
    def reward(self) -> float:
        return self._evaluate(self._get_state())["reward"]

    def is_success(self) -> bool:
        return self._evaluate(self._get_state())["success"]

    def grade(self) -> dict[str, Any]:
        return self._evaluate(self._get_state())


def _text_excerpt(html: str, limit: int = 1600) -> str:
    import re

    text = re.sub(r"(?s)<script.*?</script>", " ", html)
    text = re.sub(r"(?s)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


__all__ = ["SaltStoneEnv"]
