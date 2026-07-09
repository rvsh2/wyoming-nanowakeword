# Wyoming NanoWakeWord

A wake word server for Home Assistant that runs NanoWakeWord `.onnx` models
over the Wyoming protocol. Inference-only: it installs NanoWakeWord straight
from `arcosoph/nanowakeword` without the training extras, PyTorch, or dataset
tooling.

Ships with a ready-to-use Polish wake word **"Agata"** and a verification
pipeline that measured **0 false accepts with full recall** on unseen speech
(FLEURS pl, 2026-07-08).

## Quick start

```bash
git clone https://github.com/rvsh2/wyoming-nanowakeword.git
cd wyoming-nanowakeword
docker compose up -d --build
```

The server listens on `0.0.0.0:10400` (Wyoming) and `127.0.0.1:10401` (model
management API). In Home Assistant, add a **Wyoming** integration pointing at
the Docker host and port `10400` — the `agata` wake word appears in Voice
Assist like any openWakeWord model.

The compose file expects a whisper server on the Docker host for
[ASR verification](#asr-verification) (override with `NANOWAKEWORD_ASR_URL`
in `.env`, or remove the `--verify-asr-*` flags to run the detector alone —
then raise `threshold` to ~0.95 and expect reduced recall on unfamiliar
voices).

## How detection works

A wake word model trained on synthetic data can be sensitive or precise, but
not both — evaluating 13 trained models showed no single one with both full
recall and zero false accepts. This server splits the job:

1. **Detector** — the bundled `agata` model is a Conformer v2 trained on 29
   synthetic voices (7 Piper + 22 ElevenLabs, with "Agatka"/"Agato" as
   accepted variants). It is deliberately tuned for recall and recognises
   voices it never trained on.
2. **ASR verification** — a whisper server transcribes the candidate audio,
   and the Wyoming `Detection` is emitted only when the wake word literally
   appears in the transcript.

An idle room costs nothing: verification (~0.5 s, two transcription calls)
runs only when the detector fires.

## ASR verification

```bash
wyoming-nanowakeword ... \
  --verify-asr-url http://127.0.0.1:4050/inference \
  --verify-asr-keyword agat \
  --verify-asr-prompt "Agata? Agatka? Agato?"
```

Any whisper.cpp-compatible `/inference` endpoint works. Two passes:

1. **Unbiased pass** — the plain transcript must contain one of the
   `--verify-asr-keyword` substrings (comma-separated). An unprompted decode
   does not hallucinate the wake word into lookalike audio, so this pass
   eliminates false accepts.
2. **Prompted pass** (only with `--verify-asr-prompt`) — decoding biased
   toward the wake word must find it with mean token probability
   ≥ `--verify-asr-min-prob` (default 0.68). This recovers hard genuine
   pronunciations the unbiased pass would occasionally reject.

When the verifier is unreachable, candidates are accepted after
`verify_timeout` (fail-open) so voice control survives ASR downtime.

## Hybrid satellite + server

For satellites with weak CPUs (Raspberry/Banana Pi), run a light model
locally and let the central server confirm every candidate. Home Assistant
only ever talks to the satellite process:

```
SATELLITE DEVICE                                 CENTRAL SERVER
mic → wyoming-satellite (:10700) ← Home Assistant
        │
        ▼
      wyoming-nanowakeword            HTTP       wyoming-nanowakeword
      light model, low threshold ───────────►    strong model (+ ASR verify)
      (compose.satellite.yml)      candidate?    answers yes / no
```

- Satellite: put the light model in `./data`, set `NANOWAKEWORD_HTTP_TOKEN`
  in `.env`, then `docker compose -f compose.satellite.yml up -d`. Point it
  at the verifier with `--verify-url` / `--verify-token` / `--verify-model`
  (or from the Home Assistant integration: *Configure central verifier*).
- Central server: the regular `docker compose up -d --build`.
- Mic/speaker piece: `rhasspy/wyoming-satellite` with
  `--wake-uri tcp://127.0.0.1:10400` — see its docs for the mic/snd commands.

Rejected candidates are counted (`rejections` in `/api/scores`), published on
the event stream, and captured as `*-rejected-*.wav` for threshold tuning.
Verification here is also fail-open and can be toggled off with a switch in
Home Assistant. Both verification stages compose: the satellite verifies
against the central model, the central server gates through whisper.

## Home Assistant

- **Wyoming integration** — the only piece required: point it at the
  satellite (hybrid) or at the server (single machine).
- **Add-on (HAOS)** — add this repository as an add-on repository, install
  `nanoWakeWord`, copy models to `/share/nanowakeword`, start. All settings
  from the compose flags are available as add-on options.
- **HACS integration** (`custom_components/nanowakeword`) — a UI for the
  management API. Install this repo as a HACS custom repository (category
  *Integration*); zeroconf discovers running servers. Provides model
  upload/delete/backup/restore, *Test a recording* (score a WAV against every
  model), a real-time `Detection` event entity, per-model score sensors, and
  switches/numbers for all [runtime settings](#runtime-settings). After a
  model change it reloads the Wyoming entries, so new wake words show up in
  Voice Assist without a manual reload.

## HTTP model management API

Enabled with `--http-port` (compose and the add-on use `10401`):

- `GET /api/info` — server version and served wake words
- `GET /api/models` — models, ensembles, gates, and files
- `GET /api/scores` — live per-model scores, detection/rejection counts
- `POST /api/test?model=<id>` — score an uploaded WAV against a model
- `GET /api/events` — server-sent events stream of detections
- `POST /api/models` / `DELETE /api/models/<file>` — upload / delete
- `POST /api/refresh` — re-scan the model directory
- `GET /api/backup` / `POST /api/restore` — zip backup / restore

Uploads and restores are validated and rolled back on errors. The API can
modify files, so it never ships open to the network: compose publishes it on
`127.0.0.1`, the add-on always sets a token (auto-generated when empty — see
the add-on log). To expose it on the LAN set `--http-token <secret>`
(`NANOWAKEWORD_HTTP_TOKEN` in `.env`) and send `Authorization: Bearer`.

## Runtime settings

`GET/PATCH /api/settings` adjusts detection behavior live: `threshold`,
`trigger_level`, `refractory_seconds`, `vad_threshold`, `cascade`,
`gate_threshold`, `capture`, and the `verify_*` options. Settings persist in
`<model_dir>/settings.json` across restarts; CLI flags and add-on options act
as initial defaults. The HACS integration exposes them all as entities.

## Models

Audio in is 16 kHz / 16-bit / mono (same shape as `wyoming-openwakeword`).
Any NanoWakeWord architecture loads at runtime: `conformer`, `quartznet`,
`bcresnet`, `transformer`, `e_branchformer`, `lstm`, `gru`, `tcn`, and more —
the choice is baked in when the model is trained and exported.

Drop `.onnx` files into the model directory and describe them in
`models.yaml`. The bundled config (see `data/models.yaml`) exposes one wake
word backed by the Conformer detector at its measured operating point:

```yaml
models:
  agata:
    phrase: "Agata"
    language: "pl"
    version: "v2"
    threshold: 0.6      # candidates at 0.6 for 2 consecutive chunks,
    trigger_level: 2    # whisper makes the final call
    members:
      - model: "agata_conformer"
        role: "primary"
        threshold: 0.6

  agata_conformer:
    hidden: true        # backing model, not shown as a separate wake word
    architecture: "conformer"
    version: "v2"
```

Ensembles are supported too (`primary_and_verifier`, `weighted_average`,
`all` fusion with per-member thresholds), as is NanoWakeWord cascade mode
(`--cascade` auto-discovers `<model>_lite.onnx` gates).

## Troubleshooting

- No wake words in Home Assistant: confirm `.onnx` files exist in the model
  directory and check the logs for discovered model ids.
- False positives: enable ASR verification; or raise `threshold` /
  `trigger_level`, or enable VAD.
- Missed detections: lower `threshold`, disable VAD, or check that the
  verifier isn't rejecting genuine pronunciations (`*-rejected-*.wav`
  captures).

## References

- `arcosoph/nanowakeword` — model runtime and training
- `rhasspy/wyoming-openwakeword`, `rhasspy/wyoming-satellite`
- [Home Assistant Wyoming integration](https://www.home-assistant.io/integrations/wyoming/)
- [Home Assistant wake words](https://www.home-assistant.io/voice_control/create_wake_word/)
