# Changelog

## 0.1.0 (unreleased)

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
