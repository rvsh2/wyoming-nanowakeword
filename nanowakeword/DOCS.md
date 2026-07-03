# nanoWakeWord

Wake word detection for Home Assistant Voice Assist, running NanoWakeWord
`.onnx` models through the Wyoming protocol.

## Setup

1. Copy your `.onnx` models into `/share/nanowakeword` (or upload them later
   through the NanoWakeWord HACS integration).
2. Start the add-on.
3. In **Settings → Devices & Services**, the Wyoming integration discovers the
   server automatically; pick the wake word in your Voice Assist pipeline.

Ensembles, phrases, languages, and per-model thresholds are configured in an
optional `/share/nanowakeword/models.yaml` — see the repository README.

## Options

| Option | Description |
| ------ | ----------- |
| `model_dir` | Model directory (default `/share/nanowakeword`) |
| `default_model` | Wake word used when a client does not request one |
| `threshold` | Detection score threshold (default `0.95`) |
| `trigger_level` | Consecutive activations required before a detection |
| `refractory_seconds` | Cooldown after a detection |
| `vad_threshold` | NanoWakeWord VAD threshold, `0` disables VAD |
| `cascade` | Enable cascade mode with `<model>_lite.onnx` gate models |
| `gate_threshold` | Cascade gate threshold |
| `http_api` | Enable the model management API on port `10401` |
| `http_token` | API token; when empty, a token is generated and printed in the add-on log |
| `capture_detections` | Save a WAV of the audio around each detection to `/share/nanowakeword/captures` — training data for your next model version |
| `debug_logging` | Verbose logs |

## Model management API

With `http_api` enabled, the add-on serves a REST API on port `10401` used by
the **NanoWakeWord** HACS integration: upload models from the browser, delete,
backup/restore, live per-model scores (`/api/scores`) for threshold tuning,
testing WAV recordings against models (`/api/test`), and a detection event
stream (`/api/events`).

The API always requires a bearer token. If the `http_token` option is empty,
a persistent token is generated and printed in the add-on log — you will need
it when setting up the HACS integration.

## Runtime settings from Home Assistant

Once the NanoWakeWord HACS integration is connected, thresholds, trigger
level, refractory time, VAD, cascade, audio capture and hybrid verification
are all adjustable from Home Assistant (switch/number entities and the
integration's Configure menu). Those changes are persisted in
`/share/nanowakeword/settings.json` and win over the add-on options above on
the next start.

## Hybrid satellite + server

Satellites with weak CPUs can run a light model locally and have this add-on
confirm every candidate with its ensemble: on the satellite's integration
entry choose Configure -> "Configure central verifier" and point it at this
add-on's API (port 10401, token from the log).

## Tuning tips

- Missed detections: lower `threshold` (or the per-model/member thresholds in
  `models.yaml`).
- False triggers: raise the thresholds, raise `trigger_level`, or enable VAD.
- Use the integration's *Test a recording* action with a WAV of your wake word
  to see exactly what score every model gives before changing anything.
