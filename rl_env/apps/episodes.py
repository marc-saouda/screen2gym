"""Episode specifications shared by the simulator and the grading harness.

An *episode* is a single resettable RL sub-task. It declares:

* ``initial_overrides`` — the cumulative state the env resets to (what already
  exists at the episode's start), via three knobs:
    - ``working_sheet_rows``: ``0`` (sheet not built yet), an int, or ``None``
      meaning "all source rows" (the full-task default),
    - ``channels``: which Slack channels exist at start,
    - ``automation_level``: one of ``none | trigger | trigger_action |
      full_enabled`` (how much of the Zap is preconfigured);
* ``checks`` — named grader checks scored for reward,
* ``reward_weights`` — weight per check (reward is the normalized weighted sum),
* ``success_checks`` — the subset that must *all* pass for ``is_success()``
  (defaults to every check in ``checks``).

These defaults are used whenever a seed/curriculum does not carry its own
``episodes`` array, so the environment is self-contained. The pipeline emits the
same shape into ``outputs/curriculum.json``.
"""

from __future__ import annotations

from typing import Any, Optional

# Vocabulary of grader checks the harness knows how to evaluate.
CHECK_NAMES = (
    "working_sheet_populated",
    "channel_exists",
    "trigger_configured",
    "action_configured",
    "zap_enabled",
    "new_row_added",
    "alert_correct",
)

# The full end-to-end task (used when no episode is selected). Mirrors the
# original hard-coded grader so existing behavior is preserved.
FULL_EPISODE: dict[str, Any] = {
    "id": "full_task",
    "title": "Full booking-alert automation",
    "app": "spreadsheet",
    "goal": "Build the booking sheet, create the alerts channel, wire and enable "
            "the Zap, then validate with a new booking.",
    "initial_overrides": {
        "working_sheet_rows": None,          # all source rows present
        "channels": ["#general", "#crew-alerts"],
        "automation_level": "trigger",       # trigger preset; action incomplete; off
    },
    "checks": ["trigger_configured", "action_configured", "zap_enabled",
               "new_row_added", "alert_correct"],
    "reward_weights": {"trigger_configured": 0.10, "action_configured": 0.10,
                       "zap_enabled": 0.20, "new_row_added": 0.10,
                       "alert_correct": 0.50},
    "success_checks": ["zap_enabled", "new_row_added", "alert_correct"],
}

# The default 6-episode curriculum for the booking-alert recording.
DEFAULT_EPISODES: list[dict[str, Any]] = [
    {
        "id": "E1_create_sheet", "app": "spreadsheet",
        "title": "Create and populate the booking sheet",
        "goal": "Copy every source booking into the working sheet.",
        "initial_overrides": {"working_sheet_rows": 0, "channels": ["#general"],
                              "automation_level": "none"},
        "checks": ["working_sheet_populated"],
        "reward_weights": {"working_sheet_populated": 1.0},
    },
    {
        "id": "E2_create_channel", "app": "slack",
        "title": "Create the Slack alerts channel",
        "goal": "Create the #crew-alerts channel.",
        "initial_overrides": {"working_sheet_rows": None, "channels": ["#general"],
                              "automation_level": "none"},
        "checks": ["channel_exists"],
        "reward_weights": {"channel_exists": 1.0},
    },
    {
        "id": "E3_zap_trigger", "app": "zapier",
        "title": "Configure the Zap trigger",
        "goal": "Set the trigger to Google Sheets / New Spreadsheet Row.",
        "initial_overrides": {"working_sheet_rows": None,
                              "channels": ["#general", "#crew-alerts"],
                              "automation_level": "none"},
        "checks": ["trigger_configured"],
        "reward_weights": {"trigger_configured": 1.0},
    },
    {
        "id": "E4_zap_action", "app": "zapier",
        "title": "Configure the Zap Slack action",
        "goal": "Post a formatted New Booking Alert to #crew-alerts.",
        "initial_overrides": {"working_sheet_rows": None,
                              "channels": ["#general", "#crew-alerts"],
                              "automation_level": "trigger"},
        "checks": ["action_configured"],
        "reward_weights": {"action_configured": 1.0},
    },
    {
        "id": "E5_publish_zap", "app": "zapier",
        "title": "Publish and enable the Zap",
        "goal": "Turn the fully-configured Zap on.",
        "initial_overrides": {"working_sheet_rows": None,
                              "channels": ["#general", "#crew-alerts"],
                              "automation_level": "trigger_action"},
        "checks": ["zap_enabled"],
        "reward_weights": {"zap_enabled": 1.0},
    },
    {
        "id": "E6_validate", "app": "spreadsheet",
        "title": "Validate with a new booking",
        "goal": "Add a new booking row and confirm a correct alert posts.",
        "initial_overrides": {"working_sheet_rows": None,
                              "channels": ["#general", "#crew-alerts"],
                              "automation_level": "full_enabled"},
        "checks": ["new_row_added", "alert_correct"],
        "reward_weights": {"new_row_added": 0.4, "alert_correct": 0.6},
    },
]


def get_episodes(seed: dict) -> list[dict]:
    """Episodes carried by the seed/curriculum, or the built-in defaults."""
    eps = (seed or {}).get("episodes")
    return eps if isinstance(eps, list) and eps else DEFAULT_EPISODES


def find_episode(seed: dict, episode_id: Optional[str]) -> dict:
    """Resolve an episode by id; ``None`` selects the full-task episode."""
    if not episode_id or episode_id in ("full", "full_task"):
        return FULL_EPISODE
    for ep in get_episodes(seed):
        if ep.get("id") == episode_id:
            return ep
    raise KeyError(f"unknown episode id: {episode_id!r}")


def episode_ids(seed: dict) -> list[str]:
    return [e["id"] for e in get_episodes(seed)]
