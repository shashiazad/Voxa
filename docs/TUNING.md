# Turn-taking tuning (TurnConfig, agent/voxa_agent/config.py)

| Symptom | Knob |
|---|---|
| Agent cuts the user off mid-thought | raise `endpoint_complete_ms` (300) / `endpoint_incomplete_ms` (1500) |
| Agent feels slow to answer | lower `endpoint_complete_ms`; keep `preemptive_generation` on |
| Agent stops for noises | raise `vad_threshold_speaking` (0.65) and `stage1_min_speech_ms` (160) |
| Interrupting feels sluggish | lower `stage1_min_speech_ms`; accept more soft-pause glitches |
| Speaker echo pauses the agent | raise `vad_threshold_speaking`; ensure browser AEC; echo-text guard is automatic |
| Backchannels interrupt | add words to `extra_backchannels`; set `languages` |
| Long pauses in numbers cut off | `endpoint_incomplete_ms` up to 1500 (spec max) |

Semantic endpointing uses the LiveKit turn-detector (threshold 0.0289) plus a trailing-function-word
veto, because ASR punctuates fragments ("My order number is.") and the model trusts punctuation.

## Background sound, echo and noise (added in phase 7)

| Knob | Default | Effect |
|---|---|---|
| `echo_gate_ratio` | 0.05 | While the agent talks, mic audio below this fraction of the agent's own output level is treated as echo and cannot start a speech segment. 0 disables. |
| `near_field_gate_db` | 9 | While the agent talks, speech onsets more than this far below the caller's usual level are ignored (TV, crowd, another room). The caller's level is learned while the agent is silent. 0 disables. |
| `stage1_min_speech_ms` | 160 | Soft-pause hold. Lower = faster stop but more glitches from short noises. |

How a pause is resolved (in order): interrupt command ("wait/stop", confidence ≥ 0.4) → confirm; transcript matches our own
speech → echo, resume; words far quieter than the caller → background, resume; ≥ 2 content words → confirm; backchannel
("mm-hmm", "okay", "hi") → resume; no recognizable words for ~1.2 s → resume. With a cloud STT, a burst with no transcript at all
needs 3 s to confirm (silence from the recogniser means noise).

If callers say they "can't interrupt": check `vad_threshold_speaking` and whether the echo gate is eating a soft voice (lower
`echo_gate_ratio`). If the agent stops for the room's TV: raise `near_field_gate_db`, and consider background-voice cancellation.
