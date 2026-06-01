# Screen-Recording → RL-Environment Seed Pipeline

Given a screen recording of a person doing computer work (plus the captured
input-event stream), this pipeline outputs structured **seeds** for a
reinforcement-learning environment:

1. **What task the user was performing.**
2. **What initial-state data is needed to complete the task** — i.e. the state
   every RL episode should reset to before an agent attempts the task.
3. **A web-native, episodic curriculum** — the long session decomposed into a
   sequence of smaller, independently-resettable sub-tasks (build the sheet,
   create the channel, configure the Zap trigger / action, publish, validate),
   each with its own initial state and grader.

Two artifacts are produced: [`outputs/seed.json`](outputs/seed.json) (the
descriptive `{task, initial_state}` seed) and
[`outputs/curriculum.json`](outputs/curriculum.json) (the env-contract
`{task, initial_state, episodes[]}` the RL environment consumes directly).

This environment is intentionally **web-only**: the few desktop moments in the
recording (the WPS Office source file, a Windows Explorer open, the brief app)
are *relabeled* to web equivalents (`web_relabel.py`) so the whole trajectory
reads as websites. Desktop interaction is treated as a separate capability and
left as a documented seam for later.

---

## Why this design

A raw recording is hard for a model to reason over directly (a 1-hour 1080p
`.webm` is ~300 MB and ~100k frames). But the capture tool also emits a rich
**event stream** (`*_processed_events.jsonl`): every mouse click, keystroke,
typed string, the active window/app, the active browser URL, and pre-aligned
screenshot URLs around each action.

So the pipeline splits the problem into a cheap **deterministic** stage and a
focused **vision-language** stage:

```
recording.webm  ┐
                ├─►  (1) parse events ─►  (2) build evidence pack ─┐
events.jsonl    ┘         (no ML)            (compact digest)      │
                                                                   ▼
                          (4) VLM inference  ◄── (3) select + fetch key-frames
                                  │                  (labeled screenshots)
                                  ▼
                          seed.json  { task, initial_state }
                                  │
   (5) segment into episodes + web-relabel desktop → web, reconcile schema
                                  ▼
              curriculum.json  { task, initial_state, episodes[] }   (+ web_events.jsonl)
```

- **Deterministic stages (1–3)** never need a network or API key. They do the
  heavy lifting: reconstruct the typed text, segment the session by app,
  extract the ordered URL trail, and pick a dozen well-labeled key-frames that
  cover every phase of the session.
- **VLM stage (4)** sends that small, high-signal evidence pack + the key-frame
  images to a vision LLM, which reads exact on-screen data (table rows, the task
  brief, the Zap config) and emits the final structured seed.

This keeps token/compute cost low and makes the result reproducible and
inspectable: you can read `evidence.json` to see exactly what the model saw.

---

## Layout

| File | Role |
|------|------|
| `events.py`   | Deterministic event analysis: typed-text reconstruction, app segments, URL trail, action digest, key-frame offset selection. |
| `media.py`    | Key-frame acquisition: fetch pre-aligned remote screenshots or extract frames from the `.webm` with ffmpeg; label + base64-encode for the model. |
| `vlm.py`      | Vision-language inference: prompt construction, output schema, OpenAI-compatible (OpenAI / Azure OpenAI) call. |
| `segment.py`  | Deterministic phase detection (app / URL-group / idle) → episodes with cumulative `initial_overrides` + grader `checks`; optional VLM label enrichment. |
| `web_relabel.py` | Map non-Chrome segments to web-native (WPS source → read-only web sheet, Explorer → open URL, brief → instructions); emit a synthetic `web_events.jsonl`. |
| `curriculum.py` | Reconcile the seed into the env-contract `initial_state` shape and assemble `curriculum.json` (`task` + `initial_state` + `episodes[]`). |
| `pipeline.py` | Orchestrator + CLI; writes `evidence.json`, `seed.json`, `curriculum.json`, `web_events.jsonl`, and key-frames. |
| `outputs/`    | Generated artifacts (`evidence.json`, `seed.json`, `curriculum.json`, `web_events.jsonl`, `keyframes/`). |

---

## Usage

```bash
pip install -r requirements.txt        # Pillow (+ openai for the VLM stage)
# ffmpeg must be on PATH for video frame extraction

# Run end-to-end (downloads the video if you pass --video-url)
python pipeline.py \
  --events ../006897b3-..._processed_events.jsonl \
  --video  ../006897b3-....webm \
  --out    outputs \
  --model  gpt-4o

# To run the VLM stage, provide credentials (otherwise it is skipped and the
# evidence pack + key-frames are still written for later inference):
export OPENAI_API_KEY=sk-...
#   or Azure:  AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION

# Rebuild only the episodic curriculum from an existing seed (no VLM, no network):
python pipeline.py --events ../006897b3-..._processed_events.jsonl --out outputs --skip-vlm
#   flags:  --skip-vlm (reuse seed.json) · --no-curriculum · --label-llm (LLM episode labels)
```

**Outputs**

- `outputs/evidence.json` — the deterministic, model-ready digest (session
  summary, visited URLs, reconstructed typed text, compact action transcript,
  key-frame manifest). Always produced.
- `outputs/keyframes/` — the sampled, labeled screenshots + `manifest.json`.
- `outputs/seed.json` — the descriptive `{ task, initial_state }` seed. Produced
  by the VLM stage when credentials are available.
- `outputs/curriculum.json` — the **env-contract** `{ task, initial_state,
  episodes[] }`; loads directly into `../rl_env` (`SaltStoneEnv(seed_path=...,
  episode_id=...)`). Always produced (uses `seed.json` if present, else the env's
  example dataset as a fallback).
- `outputs/web_events.jsonl` — the event stream rewritten as web-native
  (desktop events relabeled to Chrome + a web URL, clearly marked `synthetic`).

> Note: this repo's `outputs/seed.json` is the inference result for the example
> recording. The environment used to build it had no API key, so the VLM stage
> was run by the analyzing model over the same `evidence.json` + key-frames the
> pipeline emits — i.e. exactly the inputs `vlm.py` assembles.

---

## Output schema

```jsonc
{
  "task": {
    "title", "summary", "goal", "domain",
    "applications": [...],
    "steps": [...],
    "success_criteria": [...]        // objectively checkable
  },
  "initial_state": {
    "narrative",                     // the state every episode resets to
    "starting_screen",
    "task_brief_text",
    "environment": { "os", "applications_available", "accounts_and_auth" },
    "resources": [                   // each marked GIVEN vs to_be_created
      { "name", "type", "location", "state", "details" }
    ],
    "seed_data": { "name", "source", "schema", "rows" },  // the actual records
    "preconditions": [...],
    "reset_instructions": [...]      // how to reset the env to this state
  },
  "provenance": { ... }              // session id, evidence pointers
}
```

The crucial distinction for RL: `initial_state` separates what is **GIVEN** at
`t=0` (e.g. logged-in accounts and the starter dataset) from what the agent must
**PRODUCE** (e.g. the Google Sheet, Slack channel, and the Zap), and includes
the verbatim `seed_data` so episodes can be seeded deterministically.

---

## Curriculum schema (`outputs/curriculum.json`)

The descriptive `seed.json` above and the leaner shape the RL env consumes
historically diverged; `curriculum.py` **standardizes on the env-contract
shape** and adds the episode list:

```jsonc
{
  "task": { "summary", "goal", "success_criteria", "applications", "domain" },
  "initial_state": {                  // env-contract (loads into SaltStoneEnv)
    "business_context", "starting_application",
    "datasets": [ { "name", "sheet_title", "columns", "rows": [[...]] } ],
    "accounts", "target_slack_channel", "slack_channels",
    "automation_spec": { "trigger", "action", "message_template", ... }
  },
  "episodes": [
    { "id": "E1_create_sheet", "title", "goal", "app", "time_range": [t0, t1],
      "success_criteria": [...],
      "initial_overrides": {          // the cumulative state at the episode's start
        "working_sheet_rows": 0,      // 0 (E1) | 15 (E2+) | null (all)
        "channels": ["#general"],     // which Slack channels exist yet
        "automation_level": "none"    // none | trigger | trigger_action | full_enabled
      },
      "checks": ["working_sheet_populated"],     // grader checks (RL env)
      "reward_weights": { "working_sheet_populated": 1.0 },
      "produced_artifacts": ["sheet"] }
    /* ... E2_create_channel, E3_zap_trigger, E4_zap_action, E5_publish_zap, E6_validate ... */
  ],
  "segmentation": { "n_episodes", "phases": [...] },   // deterministic boundary debug
  "web_relabel":  { "desktop_seconds", "segments": [...] }
}
```

Each episode's `initial_overrides` are exactly the knobs `rl_env` resets to, and
`checks` name the grader functions it evaluates — so the env can reset to and
grade any single episode. See `../rl_env/README.md` for the consuming side.

---

## Example result (this recording)

- **Task:** Build & publish a Zapier automation that posts real-time booking
  alerts to a Slack `#crew-alerts` channel whenever a new row is added to a
  "Salt and Stone Booking" Google Sheet (for *Salt & Stone Coastal Tours*).
- **Initial state:** Windows desktop + Postwork brief; a local
  `salt_stone_bookings_starter` spreadsheet with **15 booking records** (the
  seed data); Google/Slack/Zapier accounts already signed in; **no** Google
  Sheet, Slack channel, or Zap created yet — the agent builds those.

See [`outputs/seed.json`](outputs/seed.json) for the full structured output.
