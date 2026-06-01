"""Curriculum assembly + seed-schema reconciliation.

The VLM stage historically emitted a seed in the descriptive ``SEED_SCHEMA``
shape (``initial_state.seed_data`` / ``resources`` / ``environment``), while the
RL environment consumes a leaner "env-contract" shape
(``initial_state.datasets`` / ``target_slack_channel`` / ``automation_spec``).

This module standardizes on the **env-contract** shape and wraps it into a
``curriculum.json``:

    {
      "task":          { ... overall task ... },
      "initial_state": { ... env-contract base state ... },
      "episodes":      [ ... from segment.py ... ],
      "segmentation":  { "phases": [...] },
      "web_relabel":   { ... from web_relabel.py ... }
    }

``normalize_initial_state`` accepts either historical shape (or a partial one)
and produces the env-contract, so the same env can boot from either a generated
curriculum or a hand-written seed.
"""

from __future__ import annotations

from typing import Any, Optional

import segment
import web_relabel
from events import EventTimeline

DEFAULT_CHANNEL = "#crew-alerts"
DEFAULT_ACCOUNTS = ["google_sheets", "slack", "zapier"]
RECOMMENDED_TEMPLATE = (
    ":rotating_light: *New Booking Alert*\n"
    "*Guest Name:* {Guest_Name}\n"
    "*Group Size:* {Guest_Count}\n"
    "*Experience Type:* {Experience_Type}\n"
    "*Boat Assignment:* {Boat_Name}\n"
    "*Scheduled Start Time:* {Start_Time}\n"
    "Please prepare the boat and crew accordingly."
)


def _rows_as_lists(rows: list, columns: list[str]) -> list[list[str]]:
    """Coerce rows (list-of-dicts or list-of-lists) into list-of-lists."""
    out: list[list[str]] = []
    for r in rows or []:
        if isinstance(r, dict):
            out.append([str(r.get(c, "")) for c in columns])
        elif isinstance(r, (list, tuple)):
            out.append([str(v) for v in r])
        else:
            out.append([str(r)])
    return out


def _find_dataset(init: dict) -> dict:
    """Locate a dataset in either schema shape; return env-contract dataset."""
    datasets = init.get("datasets")
    if isinstance(datasets, list) and datasets:
        ds = dict(datasets[0])
        cols = list(ds.get("columns") or ds.get("schema") or [])
        ds["columns"] = cols
        ds["rows"] = _rows_as_lists(ds.get("rows", []), cols)
        ds.setdefault("name", "dataset")
        ds.setdefault("sheet_title", ds.get("name", "Sheet1"))
        return ds

    seed_data = init.get("seed_data")
    if isinstance(seed_data, dict):
        cols = list(seed_data.get("schema") or seed_data.get("columns") or [])
        return {
            "name": seed_data.get("name", "dataset"),
            "sheet_title": seed_data.get("sheet_title", seed_data.get("name", "Sheet1")),
            "columns": cols,
            "rows": _rows_as_lists(seed_data.get("rows", []), cols),
        }
    return {"name": "dataset", "sheet_title": "Sheet1", "columns": [], "rows": []}


def _derive_target_channel(init: dict, spec: dict) -> str:
    if init.get("target_slack_channel"):
        return init["target_slack_channel"]
    blob = " ".join(str(spec.get(k, "")) for k in ("action", "channel"))
    import re
    m = re.search(r"#([a-z0-9][a-z0-9._-]*)", blob, re.I)
    return f"#{m.group(1)}" if m else DEFAULT_CHANNEL


def normalize_initial_state(seed_like: dict,
                            dataset_fallback: Optional[dict] = None) -> dict:
    """Produce an env-contract ``initial_state`` from any seed shape."""
    init = (seed_like or {}).get("initial_state", {}) or {}
    task = (seed_like or {}).get("task", {}) or {}

    ds = _find_dataset(init)
    if not ds["rows"] and dataset_fallback:
        ds = dataset_fallback

    spec = init.get("automation_spec") or {}
    target = _derive_target_channel(init, spec)
    channels = init.get("slack_channels") or ["#general", target]

    business = (init.get("business_context") or init.get("narrative")
                or task.get("summary", ""))

    accounts = init.get("accounts")
    if not accounts:
        env = init.get("environment", {}) or {}
        accounts = env.get("accounts_and_auth") or DEFAULT_ACCOUNTS

    automation_spec = dict(spec) if spec else {
        "trigger": "Google Sheets - New Spreadsheet Row "
                   f"(sheet: '{ds.get('sheet_title')}')",
        "action": f"Slack - Send Channel Message (channel: {target}, bot 'Zapier')",
        "bot_name": "Zapier",
        "message_title": "New Booking Alert",
        "message_template": RECOMMENDED_TEMPLATE,
    }
    automation_spec.setdefault("message_template", RECOMMENDED_TEMPLATE)

    return {
        "business_context": business,
        "starting_application": init.get("starting_application", "spreadsheet"),
        "datasets": [ds],
        "accounts": accounts,
        "target_slack_channel": target,
        "slack_channels": channels,
        "automation_spec": automation_spec,
    }


def _normalize_task(seed_like: dict) -> dict:
    task = dict((seed_like or {}).get("task", {}) or {})
    task.setdefault("applications", ["spreadsheet", "slack", "zapier"])
    task.setdefault("domain", "business-automation")
    return task


def build_curriculum(timeline: EventTimeline, seed_like: dict,
                     evidence: Optional[dict] = None,
                     dataset_fallback: Optional[dict] = None,
                     label_llm: bool = False) -> dict:
    """Assemble the full web-native curriculum object."""
    initial_state = normalize_initial_state(seed_like, dataset_fallback)
    ds = initial_state["datasets"][0]
    meta = {
        "columns": ds["columns"],
        "sheet_title": ds.get("sheet_title", "Salt and Stone Booking"),
        "target_channel": initial_state["target_slack_channel"],
        "n_rows": len(ds["rows"]),
    }
    seg = segment.segment_timeline(timeline, meta, evidence=evidence,
                                   label_llm=label_llm)
    return {
        "task": _normalize_task(seed_like),
        "initial_state": initial_state,
        "episodes": seg["episodes"],
        "segmentation": {
            "n_episodes": len(seg["episodes"]),
            "phases": seg["phases"],
        },
        "web_relabel": web_relabel.relabel_report(timeline),
        "provenance": {
            "duration_seconds": round(timeline.duration, 1),
            "n_events": len(timeline.actions),
            "policy": "web-only; desktop relabeled to web; episodic curriculum",
        },
    }
