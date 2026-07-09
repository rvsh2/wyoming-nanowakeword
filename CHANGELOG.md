# Changelog

## 0.2.0 - 2026-07-09

- ASR (whisper) verification of candidate detections: `--verify-asr-url`
  points at a whisper.cpp-compatible `/inference` endpoint and the wake word
  must literally appear in the transcript before a Wyoming `Detection` is
  emitted (dual-pass rule: unbiased decode gates false accepts, optional
  prompted decode recovers hard genuine pronunciations). New settings:
  `verify_asr`, `verify_asr_url`, `verify_asr_keyword`, `verify_asr_prompt`,
  `verify_asr_min_prob`, `verify_asr_timeout`, `verify_asr_language`
- Bundled Agata model replaced with Conformer v2: trained on 29 synthetic
  voices (7 Piper + 22 ElevenLabs) with "Agatka"/"Agato" as accepted
  variants; recognises voices it never trained on. Tuned for recall and
  meant to run behind ASR verification — measured end to end on FLEURS pl:
  0 false accepts with full recall on a validated positive set
- `models.yaml`: `agata` now runs the Conformer v2 detector at
  threshold 0.6 / trigger_level 2 (the operating point measured with the
  ASR verifier); `compose.yml` wires the verifier by default
  (`NANOWAKEWORD_ASR_URL`, defaults to a whisper server on the docker host)
- Hybrid verification now sends the last 2.0 s of buffered audio (was the
  full buffer): shorter windows scored better on real captures and cut
  verification latency
- Lower hybrid-verification latency: `aiohttp` is imported at module load
  instead of on the first candidate detection, and the default
  `verify_timeout` dropped from 3.0 s to 1.0 s so an unreachable verifier
  delays wake-ups by at most one second (fail-open still applies)

## 0.1.0 - 2026-07-04

Initial release.

### Server

- Wyoming protocol wake word server for NanoWakeWord `.onnx` models
  (16 kHz / 16-bit / mono, same audio shape as wyoming-openwakeword)
- Ensemble wake words (`primary_and_verifier`, `weighted_average`, `all`
  fusion) with per-member thresholds; per-model `threshold`/`trigger_level`
  overrides in `models.yaml`
- Interpreter pooling shared across client connections, inference off the
  event loop, default wake word preloaded at startup
- Cascade mode with `<model>_lite.onnx` gate models matched per version;
  duplicate model versions resolve to the newest file
- HTTP management API (`--http-port`, bearer token): model upload/delete,
  backup/restore (validated, rolled back on errors, zip-bomb protected),
  live scores with inference times, WAV recording tests (`POST /api/test`),
  server-sent detection events, runtime settings
- Runtime settings (`GET/PATCH /api/settings`) persisted to
  `settings.json`: thresholds, trigger level, refractory, VAD, cascade,
  capture, verification — changeable from Home Assistant with no restart
- Hybrid satellite + server: `--verify-url` sends candidate audio to a
  central instance for confirmation before the Wyoming `Detection` fires;
  fail-open when the verifier is down; rejected candidates are counted and
  their audio captured (`*-rejected-*.wav`)
- Scoring (`/api/test` and hybrid verification) normalizes audio to a fixed
  peak first, so verdicts depend on content rather than microphone gain —
  level-sensitive architectures (E-Branchformer) otherwise score near zero
  on hot or quiet recordings
- The bundled Agata setup was retuned on real satellite recordings: the
  hybrid crosses quartznet (satellite) with Conformer (central verifier),
  which classified 13/13 captured real-world samples correctly
- Detection audio capture (`--capture-dir`, ring buffer, rotation) as
  training data for the next model version
- Zeroconf advertisement of both the Wyoming server and the HTTP API
- Survives an invalid model directory at startup so a broken `models.yaml`
  can be fixed remotely

### Home Assistant

- Add-on (amd64/aarch64): watchdog, AppArmor profile, auto-generated API
  token, `DOCS.md`, detection capture option
- HACS integration: config flow with zeroconf discovery and reauth; model
  upload/delete/restore and WAV testing from the browser; central verifier
  configuration; switches and numbers for all runtime settings; sensors
  (wake word models, connected clients, per-model peak scores with ensemble
  member details, last backup); real-time detection event entity; backup
  and reload buttons; services (`backup`, `restore`, `upload_model`,
  `delete_model`, `reload_models`); diagnostics; English and Polish
  translations

### Tooling

- CI: lint, types, unit tests, Home Assistant integration tests
  (pytest-homeassistant-custom-component), hassfest, HACS validation,
  add-on builds for amd64/aarch64, version consistency check
- Release: multi-arch (amd64/arm64) standalone image, add-on images tagged
  to match the add-on config version
- `compose.yml` (central, models included) and `compose.satellite.yml`
  (hybrid satellite) ready to run
