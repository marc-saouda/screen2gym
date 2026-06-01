"""End-to-end smoke test for the Salt & Stone RL environment.

Flow:
  1. reset() boots the simulated spreadsheet seeded with the 15 bookings.
  2. Configure the Zapier-style automation (channel #crew-alerts, recommended
     template, enabled).
  3. Add one new booking row.
  4. Navigate to #crew-alerts and confirm a correctly formatted "New Booking
     Alert" appeared; is_success() must be True.

Screenshots (browser mode) or HTML snapshots (headless fallback) for the initial
and success states are written to outputs/. A machine-readable report is written
to outputs/smoke_report.json.

Usage:
    python3 smoke_test.py [--backend] [--no-llm] [--headed]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.environment import SaltStoneEnv  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"

NEW_BOOKING = {
    "Booking_ID": "SST-00016",
    "Guest_Name": "Marina Holt",
    "Experience_Type": "Sunset Boat Rental",
    "Boat_Name": "Sea Breeze",
    "Guest_Count": "5",
    "Start_Time": "2024-07-18T17:30:00",
}


def _save_as(src: str | None, dst: Path) -> str | None:
    if src and Path(src).exists():
        shutil.copyfile(src, dst)
        return str(dst)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="store_true", help="force headless backend mode")
    ap.add_argument("--no-llm", action="store_true", help="disable the OpenAI judge")
    ap.add_argument("--headed", action="store_true", help="run the browser headed")
    args = ap.parse_args()

    report: dict = {"steps": [], "checks": {}, "artifacts": {}}
    env = SaltStoneEnv(
        mode="backend" if args.backend else "auto",
        headless=not args.headed,
        use_llm_judge=not args.no_llm,
    )

    ok = True
    try:
        # 1) reset / initial state ---------------------------------------
        obs = env.reset()
        report["mode"] = env.mode
        report["browser_error"] = env.browser_error
        seeded = obs["state"]["dataset"]
        n_seed = len(seeded["rows"])
        report["checks"]["seeded_15_rows"] = (n_seed == 15)
        ok &= (n_seed == 15)
        init_ext = "png" if obs["screenshot_path"] else "html"
        init_art = _save_as(
            obs["screenshot_path"] or obs["dom_snapshot_path"],
            OUTPUTS / f"initial_state.{init_ext}",
        )
        report["artifacts"]["initial_state"] = init_art
        report["steps"].append({"label": "reset", "rows": n_seed, "success": obs["success"]})
        # No alert should exist yet.
        report["checks"]["no_alert_before"] = not obs["success"]
        ok &= not obs["success"]

        # 2) configure the automation ------------------------------------
        obs = env.step(
            {
                "type": "configure_automation",
                "channel": "#crew-alerts",
                "use_recommended_template": True,
                "bot_name": "Zapier",
                "enabled": True,
            }
        )
        auto = obs["state"]["automation"]
        report["checks"]["automation_configured"] = bool(auto.get("configured"))
        report["checks"]["automation_enabled"] = bool(auto.get("enabled"))
        report["checks"]["channel_is_crew_alerts"] = (auto.get("channel_normalized") == "#crew-alerts")
        ok &= bool(auto.get("configured")) and bool(auto.get("enabled"))
        report["steps"].append({"label": "configure_automation", "reward": obs["reward"]})

        # 3) add a new booking row ---------------------------------------
        obs = env.step({"type": "add_row", "values": NEW_BOOKING})
        ds = obs["state"]["dataset"]
        report["checks"]["row_added"] = (len(ds["rows"]) == 16 and ds["new_row_count"] == 1)
        ok &= report["checks"]["row_added"]
        report["steps"].append({"label": "add_row", "rows": len(ds["rows"]), "reward": obs["reward"]})

        # 4) view #crew-alerts and grade ---------------------------------
        obs = env.step({"type": "navigate", "app": "slack", "channel": "#crew-alerts"})
        success = env.is_success()
        reward = env.reward()
        grade = env.grade()
        report["final_reward"] = reward
        report["final_success"] = success
        report["judge"] = grade["breakdown"].get("judge", {})
        report["reward_breakdown"] = grade["breakdown"].get("components", {})
        ok &= success

        alerts = [
            m
            for m in obs["state"]["slack"]["messages"]
            if m["channel"] == "#crew-alerts" and (m.get("is_bot") or m.get("source") == "automation")
        ]
        report["checks"]["alert_in_crew_alerts"] = len(alerts) >= 1
        report["alert_text"] = alerts[-1]["text"] if alerts else None
        ok &= len(alerts) >= 1

        succ_ext = "png" if obs["screenshot_path"] else "html"
        succ_art = _save_as(
            obs["screenshot_path"] or obs["dom_snapshot_path"],
            OUTPUTS / f"success_state.{succ_ext}",
        )
        report["artifacts"]["success_state"] = succ_art
        report["steps"].append({"label": "view_slack", "reward": reward, "success": success})

    except Exception as exc:  # always report, never crash silently
        import traceback

        ok = False
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        env.close()

    report["PASS"] = bool(ok)
    with open(OUTPUTS / "smoke_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _print_summary(report)
    return 0 if ok else 1


def _print_summary(report: dict) -> None:
    line = "=" * 64
    print(line)
    print("SALT & STONE RL ENV - SMOKE TEST")
    print(line)
    print(f"mode:            {report.get('mode')}")
    if report.get("browser_error"):
        print(f"browser note:    {report['browser_error']}")
    print("checks:")
    for k, v in report.get("checks", {}).items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"judge method:    {report.get('judge', {}).get('method')}")
    print(f"judge reason:    {report.get('judge', {}).get('reason', '')[:90]}")
    print(f"final reward:    {report.get('final_reward')}")
    print(f"reward parts:    {report.get('reward_breakdown')}")
    print(f"is_success():    {report.get('final_success')}")
    if report.get("alert_text"):
        print("-" * 64)
        print("alert posted to #crew-alerts:")
        print(report["alert_text"])
    print("-" * 64)
    print("artifacts:")
    for k, v in report.get("artifacts", {}).items():
        print(f"  {k}: {v}")
    if report.get("error"):
        print(f"ERROR: {report['error']}")
    print(line)
    print("RESULT:", "PASS" if report.get("PASS") else "FAIL")
    print(line)


if __name__ == "__main__":
    sys.exit(main())
