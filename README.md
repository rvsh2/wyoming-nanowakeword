# Wyoming NanoWakeWord

Wyoming NanoWakeWord is a Home Assistant wake word server that runs
NanoWakeWord `.onnx` models through the Wyoming protocol.

This project is inference-only. It installs NanoWakeWord directly from
`https://github.com/arcosoph/nanowakeword.git` and does not install the
NanoWakeWord training extras, PyTorch, notebooks, Colab helpers, or dataset
tooling.

## Audio and Models

The server follows the same audio shape used by `wyoming-openwakeword`:

- 16 kHz
- 16-bit PCM
- mono

Incoming Wyoming audio is converted with `AudioChunkConverter(rate=16000,
width=2, channels=1)`, then passed to NanoWakeWord as `np.ndarray[int16]`.
The wrapper does not build features itself. NanoWakeWord's `AudioFeatures`
handles streaming buffers internally and prepares features every 1280 samples
(80 ms at 16 kHz).

Architecture selection happens when the ONNX model is trained and exported.
At runtime this server loads any compatible NanoWakeWord `.onnx` model:
`bcresnet`, `transformer`, `conformer`, `dnn`, `lstm`, `gru`, `rnn`, `cnn`,
`tcn`, `quartznet`, `crnn`, or custom.

The bundled `agata` wake word ships a Conformer v2 detector trained on 29
synthetic voices (7 Piper + 22 ElevenLabs, including the accepted variants
"Agatka" and "Agato"). It is deliberately tuned for recall — it recognises
voices it has never heard — and is meant to run behind [ASR
verification](#asr-verification), which brings false accepts to zero.
Lesson from evaluating 13 trained models: no single synthetic-data model
achieved both full recall and zero false accepts; a sensitive detector plus
a transcription check did.

## Home Assistant Add-on

1. Add this repository as an add-on repository in Home Assistant.
2. Install the `nanoWakeWord` add-on.
3. Copy your models into `/share/nanowakeword`.
4. Start the add-on.
5. Use Home Assistant's Wyoming integration or automatic discovery to select
   the wake word in Voice Assist.

Example model directory:

```text
/share/nanowakeword/
  hey_home.onnx
  hey_home_lite.onnx
  models.yaml
```

Optional `models.yaml`:

```yaml
models:
  agata:
    phrase: "Agata"
    language: "pl"
    architecture: "ensemble:e_branchformer+transformer"
    fusion: "primary_and_verifier"
    version: "v1"
    members:
      - model: "agata_ebranchformer"
        role: "primary"
        threshold: 0.97
      - model: "agata_transformer"
        role: "verifier"
        threshold: 0.90

  agata_ebranchformer:
    hidden: true
    architecture: "e_branchformer"
    version: "v1"

  agata_transformer:
    hidden: true
    architecture: "transformer"
    version: "v1"
```

With that metadata, Home Assistant sees one wake word named `agata`, while the
server loads both `agata_ebranchformer_v1.onnx` and
`agata_transformer_v1.onnx`. A detection fires only when the primary model
passes its threshold and the verifier confirms it.

Add-on options:

- `model_dir`: model directory, default `/share/nanowakeword`
- `default_model`: model id to use when Home Assistant does not request names
- `threshold`: detection score threshold, default `0.95`
- `trigger_level`: consecutive activations required before detection
- `refractory_seconds`: cooldown after a detection
- `vad_threshold`: NanoWakeWord VAD threshold, `0` disables VAD
- `cascade`: enable NanoWakeWord cascade mode and auto-discover
  `<model>_lite.onnx`
- `gate_threshold`: cascade gate threshold
- `http_api`: enable the HTTP model management API on port `10401`
- `http_token`: bearer token for the HTTP API; when empty, the add-on
  generates a persistent token and prints it in the add-on log
- `capture_detections`: save a WAV of the audio around each detection to
  `/share/nanowakeword/captures` (training data for the next model version)
- `debug_logging`: verbose logs

## HTTP Model Management API

Started with `--http-port` (compose and the add-on enable it on `10401`), the
server exposes a small REST API for managing the model directory — the Wyoming
protocol itself cannot carry files:

- `GET /api/info` — server version and served wake words
- `GET /api/models` — models, ensembles, gates, and files
- `GET /api/scores` — live inference scores per model (last, recent peak,
  detection count, average inference time) for threshold tuning
- `POST /api/test?model=<id>` — score an uploaded WAV recording against a
  model: per-chunk score series for every ensemble member plus the fused
  score and a would-it-detect verdict
- `GET /api/events` — server-sent events stream of detections
- `POST /api/models` — multipart upload of an `.onnx` or `models.yaml`
- `DELETE /api/models/<filename>` — delete a model file
- `POST /api/refresh` — re-scan the model directory
- `GET /api/backup` — zip of the model directory
- `POST /api/restore` — replace the model directory with an uploaded zip

Uploads and restores are validated: a change that would break the model set
(for example removing an ensemble member, or an invalid `models.yaml`) is
rolled back and rejected with `400`. Only `*.onnx` files and a file named
exactly `models.yaml` are accepted, and filenames are restricted to letters,
digits, `.`, `_` and `-`.

The API can modify files, so it never ships open to the network: the add-on
always runs it with a token (auto-generated when `http_token` is empty — see
the add-on log), and compose publishes it on `127.0.0.1` only. To expose it
on the LAN, set a token (`--http-token <secret>`, or `NANOWAKEWORD_HTTP_TOKEN`
in `.env` for compose) and send `Authorization: Bearer <secret>`.

## Hybrid Satellite + Server

For satellites with weak CPUs (Raspberry Pi, Banana Pi and similar), run a
light model locally and let a strong central server confirm every candidate.
Three pieces are involved — note that Home Assistant only ever talks to the
**satellite process**, never to the wake word servers directly:

```
┌─ SATELLITE DEVICE (mic + speaker; e.g. Banana Pi) ─────────────────────────┐
│                                                                            │
│  mic → [1] wyoming-satellite (port 10700) ◄────── Home Assistant (Wyoming │
│             │ "was the wake word said?"            integration + Assist    │
│             ▼                                      pipeline)               │
│        [2] wyoming-nanowakeword — light model, low threshold, --cascade    │
│             │ candidate? send ~3 s of buffered audio for confirmation      │
└─────────────┼──────────────────────────────────────────────────────────────┘
              ▼ HTTP (LAN)
┌─ CENTRAL SERVER (strong machine) ──────────────────────────────────────────┐
│        [3] wyoming-nanowakeword — quality ensemble (POST /api/test)        │
│            answers only: "yes, that's the wake word" / "no"                │
└─────────────────────────────────────────────────────────────────────────────┘
```

The Wyoming `Detection` that wakes the Voice Assist pipeline is sent only
when the ensemble agrees. Rejected candidates are counted (`rejections` in
`/api/scores`), published on the event stream, and their audio is saved to
the capture directory (`*-rejected-*.wav`) for threshold tuning. If the
central server is unreachable, the satellite accepts candidates on its own
by default (`verify_fail_open`), so voice control survives server downtime.
Verification can be toggled off entirely with one switch in Home Assistant —
then the satellite's model decides alone.

### Running it

**Central server** (strong machine):

```bash
git clone https://github.com/rvsh2/wyoming-nanowakeword.git
cd wyoming-nanowakeword
docker compose up -d --build     # ensemble on :10400, API on :10401
```

**Satellite device** (piece [2] — the light wake word model):

```bash
# put the light model (e.g. agata_quartznet_v1.onnx + its _lite gate) in ./data
echo "NANOWAKEWORD_HTTP_TOKEN=<token>" > .env
docker compose -f compose.satellite.yml up -d    # wake on :10400, API on :10401
```

**Satellite device** (piece [1] — mic/speaker, talks to Home Assistant):

```bash
git clone https://github.com/rhasspy/wyoming-satellite.git /opt/wyoming-satellite
cd /opt/wyoming-satellite && script/setup
script/run \
  --name 'my-satellite' \
  --uri 'tcp://0.0.0.0:10700' \
  --mic-command 'parecord --rate=16000 --channels=1 --format=s16le --raw' \
  --snd-command 'paplay --rate=22050 --channels=1 --format=s16le --raw' \
  --wake-uri 'tcp://127.0.0.1:10400' \
  --wake-word-name 'agata_quartznet'
```

(Run it as a systemd service in practice; use `arecord`/`aplay` instead of
the pulse commands on ALSA-only systems.)

**Home Assistant**:

1. The **Wyoming** integration discovers the satellite (port `10700`) — pick
   it in your Voice Assist pipeline. This is the only piece HA needs.
2. Optionally add the **NanoWakeWord** integration twice — once for the
   central server and once for the satellite instance — to manage models,
   thresholds and verification from the UI.
3. Point the satellite at the verifier: satellite's integration entry →
   Configure → *Configure central verifier* (or pre-set `--verify-url`,
   `--verify-token`, `--verify-model` in `compose.satellite.yml`).

To test everything on a single machine (as a fake satellite), shift the
satellite instance's ports (e.g. `10402/10403`) so they don't collide with
the central one.

Hybrid is optional. Two simpler modes: point wyoming-satellite's
`--wake-uri` directly at the central server (no satellite container, but
audio streams over the LAN continuously), or turn the *Server verification*
switch off (the satellite's model decides alone).

## ASR Verification

A wake word model can be sensitive or precise, but synthetic training data
alone does not deliver both. ASR verification splits the job: a
high-recall detector proposes candidates, and a whisper server transcribes
the buffered audio — the detection is emitted only when the wake word
literally appears in the transcript.

```bash
wyoming-nanowakeword ... \
  --verify-asr-url http://127.0.0.1:4050/inference \
  --verify-asr-keyword agat \
  --verify-asr-prompt "Agata? Agatka? Agato?"
```

Any whisper.cpp-compatible `/inference` endpoint works (e.g. a
whisper.cpp server container). The check runs in two passes:

1. **Unbiased pass** — the transcript of the candidate audio must contain
   one of the `--verify-asr-keyword` substrings (comma-separated). An
   unprompted decode does not hallucinate the wake word into lookalike
   audio, so this pass eliminates false accepts.
2. **Prompted pass** (only when `--verify-asr-prompt` is set) — decoding
   biased toward the wake word must find it with mean token probability
   ≥ `--verify-asr-min-prob` (default 0.68). This recovers hard genuine
   pronunciations (diminutives, unusual voices) the unbiased pass alone
   would occasionally reject at the detector's operating point.

Measured with the bundled Conformer v2 detector (threshold 0.6,
trigger level 2) on Polish speech benchmarks: **0 false accepts** on
continuous unseen speech (FLEURS dev; the detector alone produced ~176
candidates/hour there, the verifier rejected every one, and it also
rejected all 450 mined false-positive segments from a separate calibration
set) with **38/38 recall** on a validated positive set that includes four
voices the detector never trained on. Verification costs two transcription
calls (~0.5 s total) per candidate — only when the detector fires, so an
idle room costs nothing.

ASR verification composes with hybrid mode: a satellite can verify against
the central model first (`--verify-url`) and the central server can gate
its own detections through whisper (`--verify-asr-url`). Both stages share
`verify_fail_open`: when a verifier is unreachable, candidates are accepted
by default so voice control survives ASR downtime.

## Runtime Settings

Detection behavior is adjustable at runtime through `GET/PATCH /api/settings`
(persisted in `<model_dir>/settings.json`, surviving restarts; CLI flags and
add-on options act as initial defaults): `threshold`, `trigger_level`,
`refractory_seconds`, `vad_threshold`, `cascade`, `gate_threshold`,
`capture`, and the `verify_*` options. The Home Assistant integration exposes
all of them as switch/number entities — no terminal needed.

## Home Assistant Integration (HACS)

The `custom_components/nanowakeword` integration is a UI for that API. Install
this repository in HACS as a custom repository (category *Integration*). When
the server runs with zeroconf, Home Assistant discovers the integration
automatically; otherwise add **NanoWakeWord** manually with the server's host
and HTTP port. It provides:

- model upload from the browser (integration options → *Upload a model file*),
  plus delete and restore flows (from an uploaded zip or from a saved backup)
- *Test a recording*: upload a WAV of your wake word (or a false trigger) and
  see the peak score of every model against its threshold
- a `Detection` event entity fired in real time on every wake word detection
  (server-sent events), usable in automations
- a `Wake word models` sensor listing served wake words, a `Connected
  clients` sensor, and per-model diagnostic `peak score` sensors (with
  ensemble member scores and inference times as attributes)
- switches (server verification, cascade, audio capture) and number entities
  (thresholds, trigger level, refractory, VAD) controlling the server's
  runtime settings — persisted server-side
- *Configure central verifier*: point a satellite at the central server for
  hybrid detection
- buttons for backup and model reload; each backup shows a notification
  with the saved path and updates a `Last backup` sensor
- services: `nanowakeword.backup` (saves a zip under `/config/nanowakeword`,
  keeps the 10 newest), `nanowakeword.restore`, `nanowakeword.upload_model`,
  `nanowakeword.delete_model`, `nanowakeword.reload_models`

After a model change the integration reloads the Wyoming config entries for
the same host, so new wake words show up in Voice Assist without a manual
reload.

## Docker Compose

The bundled `compose.yml` is ready to run, models included. The Agata `.onnx`
models ship in `data/`, so a fresh clone works with no extra setup: compose
builds the image from this repository, bind-mounts `./data`, and serves the
`agata` wake word by default — the sensitive Conformer v2 detector gated by
[ASR verification](#asr-verification). The compose file points the verifier
at a whisper server on the docker host (override with `NANOWAKEWORD_ASR_URL`
in `.env`, or delete the `--verify-asr-*` flags to run the detector alone
with a raised threshold):

```bash
git clone https://github.com/rvsh2/wyoming-nanowakeword.git
cd wyoming-nanowakeword
docker compose up -d --build
```

The server listens on `0.0.0.0:10400`. In Home Assistant, add a Wyoming
integration pointing at the Docker host IP and port `10400`; the `agata` wake
word then appears in Voice Assist exactly like an openWakeWord model.

To serve different or additional wake words, drop more `.onnx` models into
`data/`, adjust `data/models.yaml`, and change `--default-model` in
`compose.yml`. To run from a directory outside the repo instead, point the
`./data` bind mount at your own model directory.

## Troubleshooting

- No wake words in Home Assistant: confirm `.onnx` files exist in the configured
  model directory and check the add-on logs for discovered model ids.
- False positives: increase `threshold`, increase `trigger_level`, or enable VAD
  with a conservative `vad_threshold`.
- Missed detections: decrease `threshold`, disable VAD, or use a better trained
  model.

## References

- `rhasspy/wyoming-openwakeword`
- `arcosoph/nanowakeword`
- Home Assistant Wyoming integration: https://www.home-assistant.io/integrations/wyoming/
- Home Assistant wake words: https://www.home-assistant.io/voice_control/create_wake_word/
