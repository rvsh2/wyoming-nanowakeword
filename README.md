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
`tcn`, `quartznet`, `crnn`, or custom. `bcresnet` is the recommended default
architecture for new models.

- best quality candidate: `e_branchformer` (slow)
- second candidate: `conformer` (slow)
- safest NanoWakeWord candidate: `transformer` (slow)
- baseline: `quartznet` or `bcresnet` (fast)

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
- buttons for backup and model reload
- services: `nanowakeword.backup` (saves a zip under `/config/nanowakeword`,
  keeps the 10 newest), `nanowakeword.restore`, `nanowakeword.upload_model`,
  `nanowakeword.delete_model`, `nanowakeword.reload_models`

After a model change the integration reloads the Wyoming config entries for
the same host, so new wake words show up in Voice Assist without a manual
reload.

## Docker Compose

The bundled `compose.yml` is ready to run, models included. The Agata `.onnx`
models ship in `data/`, so a fresh clone works with no extra setup: compose
builds the image from this repository, bind-mounts `./data` read-only, and
serves the `agata` wake word by default. `agata` is a quality-first ensemble
defined in `data/models.yaml`: E-Branchformer (the best architecture available)
as the primary detector, confirmed by Conformer as a verifier, so a detection
fires only when both architectures agree:

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

## Polski

`wyoming-nanowakeword` to serwer Wyoming dla Home Assistant, który uruchamia
inferencję modeli NanoWakeWord `.onnx`.

Projekt jest tylko do inferencji. Instaluje NanoWakeWord bezpośrednio z
`https://github.com/arcosoph/nanowakeword.git` i nie instaluje
`nanowakeword[train]`, PyTorch, notebooków, Colab ani narzędzi datasetowych.

### Instalacja

- **Dodatek Home Assistant**: dodaj to repozytorium jako repozytorium
  dodatków, zainstaluj `nanoWakeWord`, wgraj modele do `/share/nanowakeword`
  i uruchom. Wake word wybierzesz w Voice Assist przez integrację Wyoming.
- **Docker Compose**: `docker compose up -d --build` po sklonowaniu repo —
  modele Agaty (ensemble E-Branchformer + Conformer) są w `data/` i serwer
  działa od razu na porcie `10400`.
- **HACS**: dodaj to repo jako custom repository (kategoria *Integration*)
  i zainstaluj integrację **NanoWakeWord** — daje wgrywanie modeli z
  przeglądarki, kopie zapasowe/przywracanie, sensory z liczbą modeli i
  szczytowymi score'ami (do strojenia progów) oraz serwisy
  `nanowakeword.*`. Przy włączonym zeroconf serwer jest wykrywany
  automatycznie.

### API zarządzania modelami

Serwer wystawia REST API na porcie `10401` (`--http-port`): lista modeli,
upload `.onnx`/`models.yaml`, usuwanie, podgląd score'ów (`/api/scores`),
backup jako zip i restore. Zmiany są walidowane i wycofywane, jeśli psułyby
zestaw modeli. Dodatek zawsze uruchamia API z tokenem (generowany
automatycznie i wypisywany w logu dodatku, gdy `http_token` jest pusty);
w compose API słucha tylko na `127.0.0.1`, a token ustawisz przez
`NANOWAKEWORD_HTTP_TOKEN` w `.env`.

### Strojenie progów

Sensory diagnostyczne `peak score` (per model, z score'ami członków
ensemble w atrybutach) pokazują, jak blisko progu były ostatnie próby.
Jeśli prawdziwa "Agata" nie jest wykrywana — obniż progi w `models.yaml`;
jeśli są fałszywe wyzwolenia — podnieś je albo zwiększ `trigger_level`.

## References

- `rhasspy/wyoming-openwakeword`
- `arcosoph/nanowakeword`
- Home Assistant Wyoming integration: https://www.home-assistant.io/integrations/wyoming/
- Home Assistant wake words: https://www.home-assistant.io/voice_control/create_wake_word/
