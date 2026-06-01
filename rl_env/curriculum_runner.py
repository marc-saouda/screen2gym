"""Curriculum smoke test: prove every episode is well-formed and gradable.

For each episode in the curriculum, this:

1. resets the native env and checks the goal is *not* already satisfied,
2. applies a minimal hand-written **oracle** solution and checks the episode is
   solved (``is_success()`` is True and ``reward == 1.0``),
3. applies a deliberately **wrong** attempt and checks grading discriminates it
   (``is_success()`` is False and ``reward < 1.0``),
4. repeats the oracle through ``gymnasium.make(<episode id>)`` and checks the
   episode terminates with a cumulative reward of 1.0.

Run:
    python3 curriculum_runner.py                 # all episodes, native + gym
    python3 curriculum_runner.py --episode E2_create_channel
    python3 curriculum_runner.py --llm           # use the OpenAI judge for E6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps"))

import gymnasium as gym  # noqa: E402

import gym_env  # noqa: E402
import episodes as episodes_mod  # noqa: E402
from harness.environment import SaltStoneEnv  # noqa: E402

RL_ENV_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = RL_ENV_DIR / "seed.example.json"
OUTPUTS = RL_ENV_DIR / "outputs"


def _ctx_from_seed(seed: dict) -> dict:
    init = seed.get("initial_state", {})
    ds = init.get("datasets", [{}])[0]
    cols = list(ds.get("columns", []))
    rows = [dict(zip(cols, r)) for r in ds.get("rows", [])]
    target = init.get("target_slack_channel", "#crew-alerts")
    return {"columns": cols, "source_rows": rows,
            "sheet_title": ds.get("sheet_title", "Sheet1"), "target": target}


def _new_booking(ctx: dict) -> dict:
    """A fresh booking row (id beyond the seeded ones) for the validate episode."""
    return {
        "Booking_ID": "SST-09001",
        "Guest_Name": "Marina Holt",
        "Experience_Type": "Sunset Boat Rental",
        "Boat_Name": "Sea Breeze",
        "Guest_Count": "5",
        "Start_Time": "2024-07-18T17:30:00",
    }


def oracle_actions(ep: str, ctx: dict) -> list[dict]:
    """The minimal correct solution for an episode."""
    sheet, target = ctx["sheet_title"], ctx["target"]
    if ep == "E1_create_sheet":
        return [{"type": "add_row", "values": r} for r in ctx["source_rows"]]
    if ep == "E2_create_channel":
        return [{"type": "create_channel", "name": target}]
    if ep == "E3_zap_trigger":
        return [{"type": "configure_automation", "trigger_app": "google_sheets",
                 "trigger_event": "new_row", "trigger_sheet": sheet, "enabled": False}]
    if ep == "E4_zap_action":
        return [{"type": "configure_automation", "channel": target,
                 "use_recommended_template": True, "enabled": False}]
    if ep == "E5_publish_zap":
        return [{"type": "configure_automation", "enabled": True}]
    if ep == "E6_validate":
        return [{"type": "add_row", "values": _new_booking(ctx)},
                {"type": "navigate", "app": "slack", "channel": target}]
    # full task
    return [
        {"type": "configure_automation", "channel": target,
         "use_recommended_template": True, "enabled": True},
        {"type": "add_row", "values": _new_booking(ctx)},
        {"type": "navigate", "app": "slack", "channel": target},
    ]


def wrong_actions(ep: str, ctx: dict) -> list[dict]:
    """A deliberately incorrect attempt; empty means "do nothing"."""
    if ep == "E2_create_channel":
        return [{"type": "create_channel", "name": "#wrong-channel"}]
    if ep == "E4_zap_action":
        return [{"type": "configure_automation", "channel": "#not-the-target",
                 "use_recommended_template": True, "enabled": False}]
    if ep == "E1_create_sheet":
        return [{"type": "add_row", "values": ctx["source_rows"][0]}]  # only 1 of N
    return []  # the do-nothing policy already fails every other episode


def check_native(ep: str, ctx: dict, use_llm: bool, seed_path: str,
                 mode: str = "backend", headless: bool = True) -> dict:
    env = SaltStoneEnv(seed_path=seed_path, mode=mode, headless=headless,
                       use_llm_judge=use_llm, episode_id=ep)
    actual_mode, browser_error = None, None
    try:
        obs = env.reset()
        actual_mode, browser_error = env.mode, env.browser_error
        init_success = bool(obs["success"])
        for a in oracle_actions(ep, ctx):
            env.step(a)
        g = env.grade()
        solved = bool(g["success"]) and abs(g["reward"] - 1.0) < 1e-9
    finally:
        env.close()

    envw = SaltStoneEnv(seed_path=seed_path, mode=mode, headless=headless,
                        use_llm_judge=use_llm, episode_id=ep)
    try:
        envw.reset()
        for a in wrong_actions(ep, ctx):
            envw.step(a)
        gw = envw.grade()
        discriminates = (not gw["success"]) and gw["reward"] < 1.0
    finally:
        envw.close()

    return {
        "init_not_solved": not init_success,
        "oracle_solved": solved,
        "oracle_reward": g["reward"],
        "wrong_reward": gw["reward"],
        "discriminates": discriminates,
        "mode": actual_mode,
        "browser_error": browser_error,
    }


def check_gym(ep: str, ctx: dict) -> dict:
    env = gym.make(gym_env.gym_id(ep))
    actual_mode = None
    try:
        _obs, info = env.reset()
        try:
            actual_mode = env.unwrapped.native.mode
        except Exception:
            actual_mode = None
        total, terminated = 0.0, False
        for a in oracle_actions(ep, ctx):
            _obs, r, terminated, _trunc, info = env.step(a)
            total += r
    finally:
        env.close()
    return {
        "terminated": bool(terminated),
        "cumulative_reward": round(total, 3),     # dense shaping: final - initial
        "final_reward": round(float(info.get("reward_abs", 0.0)), 3),
        "success": bool(info.get("success")),
        "mode": actual_mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=str(DEFAULT_SEED))
    ap.add_argument("--episode", default=None, help="run a single episode id")
    ap.add_argument("--llm", action="store_true", help="use the OpenAI judge (E6)")
    ap.add_argument("--include-full", action="store_true",
                    help="also run the full end-to-end task as an episode")
    ap.add_argument("--mode", default="backend", choices=["backend", "browser", "auto"],
                    help="drive the apps via real Chromium ('browser') or HTTP ('backend')")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    args = ap.parse_args()

    with open(args.seed, "r", encoding="utf-8") as fh:
        seed = json.load(fh)
    ctx = _ctx_from_seed(seed)
    headless = not args.headed
    gym_env.register_envs(args.seed, mode=args.mode, headless=headless,
                          use_llm_judge=args.llm)

    ep_ids = episodes_mod.episode_ids(seed)
    if args.include_full:
        ep_ids = ep_ids + ["full_task"]
    if args.episode:
        ep_ids = [args.episode]

    report: dict = {"seed": args.seed, "use_llm": args.llm,
                    "requested_mode": args.mode, "episodes": {}}
    all_ok = True
    want_browser = args.mode in ("browser", "auto")
    line = "=" * 86
    print(line)
    print(f"SALT & STONE CURRICULUM - PER-EPISODE ORACLE SMOKE TEST  (mode={args.mode})")
    print(line)
    header = (f"{'episode':<20}{'reset!=goal':<13}{'oracle ok':<11}"
              f"{'discrim.':<10}{'gym ok':<8}{'native':<9}{'gym':<8}")
    print(header)
    print("-" * 86)

    for ep in ep_ids:
        nat = check_native(ep, ctx, args.llm, args.seed, mode=args.mode, headless=headless)
        gymr = check_gym(ep, ctx)
        gym_ok = gymr["terminated"] and abs(gymr["final_reward"] - 1.0) < 1e-9 and gymr["success"]
        # When browser is requested, a silent fall-back to backend is a failure:
        # we are explicitly verifying the real-Chromium path here.
        mode_ok = (not args.mode == "browser") or (
            nat["mode"] == "browser" and gymr["mode"] == "browser")
        ep_ok = (nat["init_not_solved"] and nat["oracle_solved"]
                 and nat["discriminates"] and gym_ok and mode_ok)
        all_ok &= ep_ok
        report["episodes"][ep] = {"native": nat, "gym": gymr, "ok": ep_ok}
        print(f"{ep:<20}{_y(nat['init_not_solved']):<13}"
              f"{_y(nat['oracle_solved']):<11}{_y(nat['discriminates']):<10}"
              f"{_y(gym_ok):<8}{str(nat['mode']):<9}{str(gymr['mode']):<8}")

    print("-" * 86)
    # Surface any browser fall-back reason so a Chromium problem is obvious.
    for ep, r in report["episodes"].items():
        berr = r["native"].get("browser_error")
        if want_browser and r["native"]["mode"] != "browser" and berr:
            print(f"  ! {ep}: browser unavailable -> {berr}")
    report["PASS"] = bool(all_ok)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / "curriculum_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"report: {OUTPUTS / 'curriculum_report.json'}")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    print(line)
    return 0 if all_ok else 1


def _y(v: bool) -> str:
    return "PASS" if v else "FAIL"


if __name__ == "__main__":
    sys.exit(main())
