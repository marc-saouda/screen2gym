"""In-memory state model and automation engine for the simulated apps.

The state is intentionally simple and fully resettable: a fresh copy is rebuilt
from a seed dict on every reset, so RL episodes are deterministic.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


def normalize_channel(name: str) -> str:
    """Normalize a Slack channel reference to the '#name' form."""
    name = (name or "").strip().lower()
    if not name:
        return ""
    name = name.lstrip("#").strip()
    name = re.sub(r"\s+", "-", name)
    return f"#{name}" if name else ""


DEFAULT_TEMPLATE = (
    ":rotating_light: *New Booking Alert*\n"
    "*Guest Name:* {Guest_Name}\n"
    "*Group Size:* {Guest_Count}\n"
    "*Experience Type:* {Experience_Type}\n"
    "*Boat Assignment:* {Boat_Name}\n"
    "*Scheduled Start Time:* {Start_Time}\n"
    "Please prepare the boat and crew accordingly."
)


@dataclass
class Automation:
    """The simulated Zapier zap: a Sheets trigger wired to a Slack action.

    The trigger and action are tracked independently so a curriculum can grade
    "configure the trigger" and "configure the action" as separate episodes.
    """

    trigger_app: str = ""
    trigger_event: str = ""
    trigger_sheet: str = ""
    action_app: str = ""
    action_event: str = ""
    channel: str = ""
    bot_name: str = "Zapier"
    message_template: str = ""
    enabled: bool = False

    def trigger_configured(self) -> bool:
        return (
            self.trigger_app == "google_sheets"
            and self.trigger_event == "new_row"
            and bool(str(self.trigger_sheet).strip())
        )

    def action_configured(self) -> bool:
        return (
            self.action_app == "slack"
            and self.action_event == "send_channel_message"
            and bool(normalize_channel(self.channel))
            and bool(str(self.message_template).strip())
        )

    def is_configured(self) -> bool:
        return self.trigger_configured() and self.action_configured()

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["trigger_is_configured"] = self.trigger_configured()
        d["action_is_configured"] = self.action_configured()
        d["configured"] = self.is_configured()
        d["channel_normalized"] = normalize_channel(self.channel)
        return d


def automation_for_level(level: str, spec: dict, sheet_title: str,
                         target: str, recommended: str) -> "Automation":
    """Build the automation preconfigured to ``level``.

    none           -> nothing set (agent configures trigger + action)
    trigger        -> trigger set; action app/event preset but channel/template empty
    trigger_action -> trigger + action fully set (channel=target, template); off
    full_enabled   -> everything set + enabled
    """
    bot = (spec or {}).get("bot_name", "Zapier")
    auto = Automation(bot_name=bot)
    if level == "none":
        return auto
    auto.trigger_app = "google_sheets"
    auto.trigger_event = "new_row"
    auto.trigger_sheet = sheet_title
    auto.action_app = "slack"
    auto.action_event = "send_channel_message"
    if level == "trigger":
        return auto
    auto.channel = target
    auto.message_template = recommended
    if level == "full_enabled":
        auto.enabled = True
    return auto


@dataclass
class SlackMessage:
    channel: str
    author: str
    text: str
    ts: float
    is_bot: bool = False
    source: str = "user"  # "user" or "automation"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SimState:
    """Full simulator state, rebuilt from a seed on every reset."""

    seed: dict[str, Any]
    business_context: str = ""
    task_summary: str = ""
    task_goal: str = ""
    success_criteria: list[str] = field(default_factory=list)
    dataset_name: str = "dataset"
    sheet_title: str = "Sheet1"
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    seed_row_count: int = 0
    # Read-only source data the working sheet is copied from (web-relabeled WPS).
    source_columns: list[str] = field(default_factory=list)
    source_rows: list[dict[str, str]] = field(default_factory=list)
    slack_channels: list[str] = field(default_factory=list)
    messages: list[SlackMessage] = field(default_factory=list)
    automation: Automation = field(default_factory=Automation)
    automation_spec: dict[str, Any] = field(default_factory=dict)
    target_channel: str = "#crew-alerts"
    episode_id: Optional[str] = None
    initial_overrides: dict[str, Any] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ----- construction -------------------------------------------------
    @classmethod
    def from_seed(cls, seed: dict[str, Any],
                  episode_id: Optional[str] = None) -> "SimState":
        import episodes as episodes_mod  # local import: same package dir

        task = seed.get("task", {})
        init = seed.get("initial_state", {})
        datasets = init.get("datasets", [])
        ds = datasets[0] if datasets else {"columns": [], "rows": []}
        columns = list(ds.get("columns", []))
        source_rows = [dict(zip(columns, r)) for r in ds.get("rows", [])]
        spec = init.get("automation_spec", {})
        target = normalize_channel(init.get("target_slack_channel", "#crew-alerts")) or "#crew-alerts"
        sheet_title = ds.get("sheet_title", ds.get("name", "Sheet1"))

        episode = episodes_mod.find_episode(seed, episode_id)
        overrides = dict(episode.get("initial_overrides", {}))

        # --- working sheet rows: 0, an int, or None (= all source rows) -----
        n_override = overrides.get("working_sheet_rows", None)
        if n_override is None:
            working_rows = [dict(r) for r in source_rows]
        else:
            n = max(0, min(int(n_override), len(source_rows)))
            working_rows = [dict(r) for r in source_rows[:n]]

        # --- channels that exist at episode start ---------------------------
        ch_override = overrides.get("channels")
        if ch_override is not None:
            channels = [normalize_channel(c) for c in ch_override]
        else:
            channels = [normalize_channel(c) for c in (init.get("slack_channels") or ["#general", target])]
        channels = [c for c in channels if c]
        if "#general" not in channels:
            channels.insert(0, "#general")
        # The full/default task always exposes the target channel (legacy behavior);
        # named episodes control channel existence explicitly (e.g. E2 creates it).
        if (episode_id in (None, "full", "full_task")) and target not in channels:
            channels.append(target)
        seen: set[str] = set()
        channels = [c for c in channels if not (c in seen or seen.add(c))]

        recommended = (spec or {}).get("message_template") or DEFAULT_TEMPLATE

        state = cls(
            seed=copy.deepcopy(seed),
            business_context=init.get("business_context", ""),
            task_summary=task.get("summary", ""),
            task_goal=task.get("goal", ""),
            success_criteria=list(task.get("success_criteria", [])),
            dataset_name=ds.get("name", "dataset"),
            sheet_title=sheet_title,
            columns=columns,
            rows=working_rows,
            seed_row_count=len(working_rows),
            source_columns=columns,
            source_rows=source_rows,
            slack_channels=channels,
            automation_spec=spec,
            target_channel=target,
            episode_id=episode.get("id"),
            initial_overrides=overrides,
        )
        level = overrides.get("automation_level", "trigger")
        state.automation = automation_for_level(level, spec, sheet_title, target, recommended)

        # A little seeded chatter in #general so the Slack view looks real.
        now = time.time()
        state.messages.append(
            SlackMessage(
                channel="#general",
                author="dockmaster",
                text="Morning crew - check the board for today's departures.",
                ts=now - 3600,
                is_bot=False,
                source="user",
            )
        )
        return state

    # ----- recommended template ----------------------------------------
    def recommended_template(self) -> str:
        tpl = (self.automation_spec or {}).get("message_template", "")
        return tpl or DEFAULT_TEMPLATE

    # ----- mutations ----------------------------------------------------
    def add_row(self, values: dict[str, str]) -> dict[str, Any]:
        """Append a booking row; fire the automation if it is live."""
        with self.lock:
            row = {c: str(values.get(c, "")).strip() for c in self.columns}
            self.rows.append(row)
            self.event_log.append({"t": time.time(), "type": "row_added", "row": row})
            posted = self._maybe_fire_automation(row)
            return {"row": row, "posted_message": posted}

    def _maybe_fire_automation(self, row: dict[str, str]) -> Optional[dict[str, Any]]:
        auto = self.automation
        if not (auto.enabled and auto.is_configured()):
            return None
        text = self.render_template(auto.message_template, row)
        channel = normalize_channel(auto.channel)
        msg = SlackMessage(
            channel=channel,
            author=auto.bot_name or "Zapier",
            text=text,
            ts=time.time(),
            is_bot=True,
            source="automation",
        )
        self.messages.append(msg)
        self.event_log.append(
            {"t": msg.ts, "type": "automation_posted", "channel": channel, "row": row}
        )
        return msg.to_dict()

    @staticmethod
    def render_template(template: str, row: dict[str, str]) -> str:
        """Substitute {Column} and {{Column}} placeholders with row values."""
        def repl(match: re.Match) -> str:
            key = match.group(1).strip()
            return str(row.get(key, match.group(0)))

        text = re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", repl, template)
        text = re.sub(r"\{\s*([^{}]+?)\s*\}", repl, text)
        return text

    def update_automation(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            auto = self.automation
            for key in (
                "trigger_app",
                "trigger_event",
                "trigger_sheet",
                "action_app",
                "action_event",
                "channel",
                "bot_name",
                "message_template",
            ):
                if key in patch and patch[key] is not None:
                    setattr(auto, key, str(patch[key]))
            if "enabled" in patch and patch["enabled"] is not None:
                val = patch["enabled"]
                if isinstance(val, str):
                    val = val.strip().lower() in ("1", "true", "on", "yes")
                auto.enabled = bool(val)
            self.event_log.append({"t": time.time(), "type": "automation_updated", "config": auto.to_dict()})
            return auto.to_dict()

    def create_channel(self, name: str) -> dict[str, Any]:
        """Create a Slack channel if it does not already exist."""
        with self.lock:
            ch = normalize_channel(name)
            created = False
            if ch and ch not in self.slack_channels:
                self.slack_channels.append(ch)
                created = True
                self.event_log.append({"t": time.time(), "type": "channel_created", "channel": ch})
            return {"channel": ch, "created": created, "channels": list(self.slack_channels)}

    def post_message(self, channel: str, text: str, author: str = "you") -> dict[str, Any]:
        with self.lock:
            msg = SlackMessage(
                channel=normalize_channel(channel),
                author=author,
                text=text,
                ts=time.time(),
                is_bot=False,
                source="user",
            )
            self.messages.append(msg)
            return msg.to_dict()

    # ----- views --------------------------------------------------------
    def messages_for(self, channel: str) -> list[SlackMessage]:
        ch = normalize_channel(channel)
        return [m for m in self.messages if m.channel == ch]

    def new_rows(self) -> list[dict[str, str]]:
        return self.rows[self.seed_row_count :]

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "business_context": self.business_context,
                "episode": {
                    "id": self.episode_id,
                    "initial_overrides": self.initial_overrides,
                },
                "task": {
                    "summary": self.task_summary,
                    "goal": self.task_goal,
                    "success_criteria": self.success_criteria,
                },
                "dataset": {
                    "name": self.dataset_name,
                    "sheet_title": self.sheet_title,
                    "columns": self.columns,
                    "rows": self.rows,
                    "seed_row_count": self.seed_row_count,
                    "new_row_count": len(self.rows) - self.seed_row_count,
                },
                "source": {
                    "columns": self.source_columns,
                    "rows": self.source_rows,
                    "read_only": True,
                },
                "slack": {
                    "channels": self.slack_channels,
                    "target_channel": self.target_channel,
                    "messages": [m.to_dict() for m in self.messages],
                },
                "automation": self.automation.to_dict(),
                "event_log": self.event_log,
            }


def load_seed(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
