"""End-to-end pipeline: screen recording -> RL-environment seed.

Stages
------
1. **Parse events**       (``events.EventTimeline``)  — deterministic.
2. **Build evidence**     — compact, model-ready digest of the session.
3. **Select key-frames**  (``media``)                 — labeled screenshots.
4. **VLM inference**      (``vlm``)                    — task + initial_state.
5. **Write outputs**      — ``evidence.json``, ``seed.json``, key-frames,
                            and a ``keyframes/manifest.json``.

The deterministic stages always run (no network, no API key). The VLM stage
runs only when credentials are available; otherwise the evidence pack and
key-frames are still written so inference can be performed afterwards.

Usage
-----
    python pipeline.py \
        --events  ../006897b3-..._processed_events.jsonl \
        --video   ../006897b3-....webm \
        --out     outputs \
        --model   gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import curriculum as curriculum_mod
import media
import vlm
import web_relabel
from events import EventTimeline

# Used as a last-resort dataset source so the curriculum always carries the
# seed rows even if VLM inference was skipped and no seed.json exists yet.
_ENV_SEED_FALLBACK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "rl_env", "seed.example.json")


def _load_seed_like(out_dir: str, inferred: dict | None) -> dict:
    """Pick the best available seed-like object for curriculum assembly."""
    if inferred:
        return inferred
    for path in (os.path.join(out_dir, "seed.json"), _ENV_SEED_FALLBACK):
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    return json.load(fh)
            except Exception:
                pass
    return {}


def _dataset_fallback() -> dict | None:
    """The env's example dataset (15 booking rows) as a fallback."""
    if os.path.exists(_ENV_SEED_FALLBACK):
        try:
            with open(_ENV_SEED_FALLBACK) as fh:
                seed = json.load(fh)
            return curriculum_mod.normalize_initial_state(seed)["datasets"][0]
        except Exception:
            return None
    return None


def build_evidence(timeline: EventTimeline, keyframes: list) -> dict:
    """Assemble the deterministic, model-ready evidence pack."""
    return {
        "summary": timeline.summary(),
        "brief_text": "",  # filled by VLM from the first frame, or via --brief-text
        "visited_urls": timeline.visited_urls(),
        "typed_texts": timeline.typed_texts(min_len=2),
        "action_digest": timeline.action_digest(max_lines=160),
        "app_segments": [s.to_dict() for s in timeline.app_segments()],
        "keyframes": [k.to_dict() for k in keyframes],
    }


def maybe_download(url: str, dest: str) -> str:
    if os.path.exists(dest):
        return dest
    print(f"[pipeline] downloading video -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def run(events_path: str, out_dir: str, video_path: str | None = None,
        video_url: str | None = None, model: str = "gpt-4o",
        max_frames: int = 12, brief_text: str | None = None,
        skip_vlm: bool = False, build_curriculum: bool = True,
        label_llm: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    kf_dir = os.path.join(out_dir, "keyframes")
    # Ensure the key-frame dir exists even when --skip-vlm means media.materialize
    # (which would otherwise create it) is never called, so manifest.json can write.
    os.makedirs(kf_dir, exist_ok=True)

    if video_url and not video_path:
        video_path = maybe_download(
            video_url, os.path.join(out_dir, "recording.webm"))

    print("[pipeline] parsing events ...")
    timeline = EventTimeline.from_jsonl(events_path)
    print(f"[pipeline]   {len(timeline.actions)} actions, "
          f"{timeline.duration:.0f}s, apps={timeline.summary()['applications']}")

    print("[pipeline] selecting key-frames ...")
    keyframes = media.select_keyframes(timeline, max_frames=max_frames)
    if skip_vlm:
        # Curriculum building needs only the timeline; skip image downloads.
        print(f"[pipeline]   {len(keyframes)} key-frames selected (images not "
              "materialized: --skip-vlm)")
    else:
        keyframes = media.materialize(keyframes, kf_dir, video_path=video_path)
        got = sum(1 for k in keyframes if k.path)
        print(f"[pipeline]   {got}/{len(keyframes)} key-frames available")

    evidence = build_evidence(timeline, keyframes)
    if brief_text:
        evidence["brief_text"] = brief_text

    with open(os.path.join(out_dir, "evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2)
    with open(os.path.join(kf_dir, "manifest.json"), "w") as fh:
        json.dump([k.to_dict() for k in keyframes], fh, indent=2)
    print(f"[pipeline]   wrote evidence.json + keyframes/manifest.json")

    seed = None
    if skip_vlm:
        print("[pipeline] skipping VLM inference (--skip-vlm)")
    else:
        print("[pipeline] running VLM inference ...")
        seed = vlm.infer_seed(evidence, [k.to_dict() for k in keyframes], model=model)
        if seed is None:
            print("[pipeline]   no LLM credentials (set OPENAI_API_KEY or "
                  "AZURE_OPENAI_*). Skipped inference.")
            print("[pipeline]   -> evidence pack + key-frames are ready for any VLM.")
        else:
            with open(os.path.join(out_dir, "seed.json"), "w") as fh:
                json.dump(seed, fh, indent=2)
            print("[pipeline]   wrote seed.json")

    curriculum = None
    if build_curriculum:
        print("[pipeline] building episodic web-native curriculum ...")
        seed_like = _load_seed_like(out_dir, seed)
        curriculum = curriculum_mod.build_curriculum(
            timeline, seed_like, evidence=evidence,
            dataset_fallback=_dataset_fallback(), label_llm=label_llm)
        with open(os.path.join(out_dir, "curriculum.json"), "w") as fh:
            json.dump(curriculum, fh, indent=2)
        n_ev = web_relabel.write_web_events(
            timeline, os.path.join(out_dir, "web_events.jsonl"))
        eps = curriculum["episodes"]
        print(f"[pipeline]   wrote curriculum.json ({len(eps)} episodes: "
              f"{', '.join(e['id'] for e in eps)})")
        print(f"[pipeline]   wrote web_events.jsonl ({n_ev} events, "
              f"{curriculum['web_relabel']['desktop_seconds']}s desktop relabeled)")

    return {"evidence": evidence, "seed": seed, "curriculum": curriculum}


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen recording -> RL seed pipeline")
    ap.add_argument("--events", required=True, help="processed_events.jsonl path")
    ap.add_argument("--video", default=None, help="local .webm path (optional)")
    ap.add_argument("--video-url", default=None, help="download video if no local")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--model", default="gpt-4o", help="vision model name")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--brief-text", default=None,
                    help="optional: on-screen task brief text if known")
    ap.add_argument("--skip-vlm", action="store_true",
                    help="skip VLM inference; (re)build curriculum from existing seed")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="do not build the episodic curriculum.json")
    ap.add_argument("--label-llm", action="store_true",
                    help="use an LLM to rewrite episode titles/goals/criteria")
    args = ap.parse_args()
    run(args.events, args.out, video_path=args.video, video_url=args.video_url,
        model=args.model, max_frames=args.max_frames, brief_text=args.brief_text,
        skip_vlm=args.skip_vlm, build_curriculum=not args.no_curriculum,
        label_llm=args.label_llm)


if __name__ == "__main__":
    main()
