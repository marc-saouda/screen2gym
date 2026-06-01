"""Action schema for the Salt & Stone RL environment.

Actions are plain dicts so they are trivial to serialize for an RL policy. Two
tiers are supported:

High-level (semantic) actions, applied identically in browser and headless modes:
  {"type": "navigate", "app": "spreadsheet"|"slack"|"zapier", "channel": "#crew-alerts"}
  {"type": "add_row", "values": {"Guest_Name": "...", ...}}
  {"type": "configure_automation", "channel": "#crew-alerts",
      "message_template": "...", "bot_name": "Zapier", "enabled": true,
      "use_recommended_template": false}
  {"type": "screenshot"}
  {"type": "noop"}

Low-level (browser-only) actions, for agents that operate on the DOM directly:
  {"type": "click", "selector": "#add-row-btn"}
  {"type": "fill", "selector": "#zap-channel", "text": "#crew-alerts"}
"""
from __future__ import annotations

from typing import Any

VALID_TYPES = {
    "navigate",
    "add_row",
    "create_channel",
    "configure_automation",
    "screenshot",
    "noop",
    "click",
    "fill",
}

BROWSER_ONLY = {"click", "fill"}


def navigate(app: str, channel: str | None = None) -> dict[str, Any]:
    action: dict[str, Any] = {"type": "navigate", "app": app}
    if channel:
        action["channel"] = channel
    return action


def add_row(values: dict[str, str] | None = None) -> dict[str, Any]:
    return {"type": "add_row", "values": values or {}}


def create_channel(name: str = "#crew-alerts") -> dict[str, Any]:
    return {"type": "create_channel", "name": name}


def configure_automation(
    channel: str | None = None,
    message_template: str | None = None,
    bot_name: str = "Zapier",
    enabled: bool = True,
    use_recommended_template: bool = False,
    trigger_app: str | None = None,
    trigger_event: str | None = None,
    trigger_sheet: str | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "type": "configure_automation",
        "bot_name": bot_name,
        "enabled": enabled,
        "use_recommended_template": use_recommended_template,
    }
    # Only include fields that were explicitly provided so callers can configure
    # the trigger and the action independently (per-episode) without clobbering.
    if channel is not None:
        action["channel"] = channel
    if message_template is not None:
        action["message_template"] = message_template
    if trigger_app is not None:
        action["trigger_app"] = trigger_app
    if trigger_event is not None:
        action["trigger_event"] = trigger_event
    if trigger_sheet is not None:
        action["trigger_sheet"] = trigger_sheet
    return action


def click(selector: str) -> dict[str, Any]:
    return {"type": "click", "selector": selector}


def fill(selector: str, text: str) -> dict[str, Any]:
    return {"type": "fill", "selector": selector, "text": text}


def screenshot() -> dict[str, Any]:
    return {"type": "screenshot"}


def validate(action: dict[str, Any]) -> None:
    if not isinstance(action, dict) or "type" not in action:
        raise ValueError("action must be a dict with a 'type' key")
    if action["type"] not in VALID_TYPES:
        raise ValueError(f"unknown action type: {action['type']!r}")
