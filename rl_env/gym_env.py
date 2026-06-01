"""Thin Gymnasium adapter over the native :class:`SaltStoneEnv`.

The native environment already exposes ``reset`` / ``step`` / ``reward`` /
``is_success``, but with a richer (dict) observation and a semantic dict action.
This module wraps it in the standard Gymnasium API so the curriculum can be
driven by any RL library:

    import gym_env
    gym_env.register_envs()                      # registers one id per episode
    env = gymnasium.make("SaltStone/E2_create_channel-v0")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(
        '{"type": "create_channel", "name": "#crew-alerts"}')

Spaces (intentionally simple, since the policy here is typically an LLM):

* observation: ``Dict`` of ``Text`` fields (url, page_text, state_json, ...),
* action: a single ``Text`` field holding a JSON-encoded semantic action; the
  ``step`` method also accepts a native action dict directly.

Reward is the *change* in the episode's shaped score per step (so the return
over an episode equals the final shaped score in ``[0, 1]``); the absolute score
and the full grader breakdown are returned in ``info``. ``terminated`` is the
episode's success condition; ``truncated`` fires at ``max_steps``.
"""

from __future__ import annotations

import json
import os
import string
import sys
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces

RL_ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RL_ENV_DIR))
sys.path.insert(0, str(RL_ENV_DIR / "apps"))

from harness.environment import SaltStoneEnv  # noqa: E402
import episodes as episodes_mod  # noqa: E402

DEFAULT_SEED = RL_ENV_DIR / "seed.example.json"
_CHARSET = frozenset(string.printable)
_PAGE_MAX = 8000
_STATE_MAX = 20000
_ACTION_MAX = 4096


class SaltStoneGymEnv(gym.Env):
    """Gymnasium wrapper around a single curriculum episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_id: Optional[str] = None,
        seed_path: str | os.PathLike = DEFAULT_SEED,
        mode: str = "backend",
        use_llm_judge: bool = False,
        max_steps: int = 40,
        render_mode: Optional[str] = None,
        **env_kwargs: Any,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self._native_kwargs = dict(
            seed_path=str(seed_path), mode=mode,
            use_llm_judge=use_llm_judge, **env_kwargs,
        )
        self.native = SaltStoneEnv(episode_id=episode_id, **self._native_kwargs)
        self._prev_reward = 0.0
        self._steps = 0

        self.observation_space = spaces.Dict({
            "url": spaces.Text(max_length=1024, charset=_CHARSET),
            "screenshot_path": spaces.Text(max_length=1024, charset=_CHARSET),
            "page_text": spaces.Text(max_length=_PAGE_MAX, charset=_CHARSET),
            "state_json": spaces.Text(max_length=_STATE_MAX, charset=_CHARSET),
            "episode": spaces.Text(max_length=64, charset=_CHARSET),
        })
        # A semantic action encoded as JSON text (step() also accepts a dict).
        self.action_space = spaces.Text(max_length=_ACTION_MAX, charset=_CHARSET)

    # ----- helpers ------------------------------------------------------
    @staticmethod
    def _decode_action(action: Any) -> dict:
        if isinstance(action, dict):
            return action
        if isinstance(action, (bytes, bytearray)):
            action = action.decode("utf-8", "ignore")
        if isinstance(action, str):
            action = action.strip()
            if not action:
                return {"type": "noop"}
            try:
                parsed = json.loads(action)
                return parsed if isinstance(parsed, dict) else {"type": "noop"}
            except json.JSONDecodeError:
                return {"type": "noop"}
        return {"type": "noop"}

    def _encode_obs(self, o: dict) -> dict:
        state = o.get("state", {})
        page = o.get("html_excerpt") or ""
        if not page and o.get("aria"):
            page = json.dumps(o["aria"], default=str)
        ep = self.native.episode_id or "full_task"
        return {
            "url": str(o.get("url", ""))[:1024],
            "screenshot_path": str(o.get("screenshot_path") or "")[:1024],
            "page_text": str(page)[:_PAGE_MAX],
            "state_json": json.dumps(state, default=str)[:_STATE_MAX],
            "episode": ep[:64],
        }

    def _info(self, o: dict) -> dict:
        return {
            "reward_abs": float(o.get("reward", 0.0)),
            "success": bool(o.get("success")),
            "breakdown": o.get("reward_breakdown", {}),
            "applied": (o.get("info") or {}).get("applied"),
            "episode": self.native.episode_id or "full_task",
            "mode": o.get("mode"),
        }

    # ----- gym API ------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        new_ep = (options or {}).get("episode_id", "__keep__")
        if new_ep != "__keep__" and new_ep != self.native.episode_id:
            self.native.close()
            self.native = SaltStoneEnv(episode_id=new_ep, **self._native_kwargs)
        o = self.native.reset()
        self._prev_reward = float(o.get("reward", 0.0))
        self._steps = 0
        return self._encode_obs(o), self._info(o)

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        o = self.native.step(self._decode_action(action))
        reward_abs = float(o.get("reward", 0.0))
        delta = reward_abs - self._prev_reward
        self._prev_reward = reward_abs
        self._steps += 1
        terminated = bool(o.get("success"))
        truncated = self._steps >= self.max_steps
        return self._encode_obs(o), float(delta), terminated, truncated, self._info(o)

    def close(self) -> None:
        try:
            self.native.close()
        except Exception:
            pass


def gym_id(episode_id: Optional[str]) -> str:
    if episode_id in (None, "full", "full_task"):
        return "SaltStone/Full-v0"
    return f"SaltStone/{episode_id}-v0"


def register_envs(seed_path: str | os.PathLike = DEFAULT_SEED,
                  **env_kwargs: Any) -> list[str]:
    """Register one Gym id per episode (+ the full task). Returns the ids."""
    with open(seed_path, "r", encoding="utf-8") as fh:
        seed = json.load(fh)
    ids: list[str] = []
    targets: list[Optional[str]] = [None] + episodes_mod.episode_ids(seed)
    for ep in targets:
        env_id = gym_id(ep)
        if env_id in gym.envs.registry:  # idempotent re-registration
            del gym.envs.registry[env_id]
        gym.register(
            id=env_id,
            entry_point=SaltStoneGymEnv,
            disable_env_checker=True,
            kwargs={"episode_id": ep, "seed_path": str(seed_path), **env_kwargs},
        )
        ids.append(env_id)
    return ids


__all__ = ["SaltStoneGymEnv", "register_envs", "gym_id"]
