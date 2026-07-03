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
architecture for new models, but it is not a runtime switch.

For quality-first GPU-trained models, especially when training on hardware like
2x RTX 3090 and runtime CPU size is not the main constraint, benchmark multiple
architectures instead of choosing a small recurrent model first:

- best quality candidate: `e_branchformer`
- second candidate: `conformer`
- safest NanoWakeWord candidate: `transformer`
- baseline: `quartznet` or `bcresnet`

For the wake word `Agata`, the recommended production shape is an ensemble:
`E-Branchformer` as the primary model and `Transformer` as the verifier.

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
- `debug_logging`: verbose logs

## HTTP Model Management API

Started with `--http-port` (compose and the add-on enable it on `10401`), the
server exposes a small REST API for managing the model directory — the Wyoming
protocol itself cannot carry files:

- `GET /api/info` — server version and served wake words
- `GET /api/models` — models, ensembles, gates, and files
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
this repository in HACS as a custom repository (category *Integration*), then
add the **NanoWakeWord** integration pointing at the server's host and HTTP
port. It provides:

- model upload from the browser (integration options → *Upload a model file*),
  plus delete and backup-restore flows
- a `Wake word models` sensor listing served wake words, and buttons for
  backup and model reload
- services: `nanowakeword.backup` (saves a zip under `/config/nanowakeword`),
  `nanowakeword.restore`, `nanowakeword.upload_model`,
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
docker compose -f compose.yml up -d --build
```

The server listens on `0.0.0.0:10400`. In Home Assistant, add a Wyoming
integration pointing at the Docker host IP and port `10400`; the `agata` wake
word then appears in Voice Assist exactly like an openWakeWord model.

To serve different or additional wake words, drop more `.onnx` models into
`data/`, adjust `data/models.yaml`, and change `--default-model` in
`compose.yml`. To run from a directory outside the repo instead, point the
`./data` bind mount at your own model directory.

## Standalone Python

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
wyoming-nanowakeword \
  --uri tcp://0.0.0.0:10400 \
  --model-dir /opt/wyoming-nanowakeword/share/nanowakeword \
  --zeroconf nanoWakeWord
```

## Troubleshooting

- No wake words in Home Assistant: confirm `.onnx` files exist in the configured
  model directory and check the add-on logs for discovered model ids.
- Cascade does not activate: make sure `<model>_lite.onnx` is next to
  `<model>.onnx`.
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

Najważniejsze:

- audio wejściowe jest konwertowane do `16 kHz`, `16-bit`, mono;
- do NanoWakeWord trafiają surowe próbki `np.int16`;
- wrapper nie tworzy własnych feature'ów;
- architektura modelu jest częścią pliku ONNX, a nie opcją runtime;
- przy treningu quality-first dla `Agata` rekomendowany jest benchmark:
  `e_branchformer`, `conformer`, `transformer` oraz baseline `quartznet` albo
  `bcresnet`;
- produkcyjnie dla `Agata` warto użyć ensemble: E-Branchformer jako primary i
  Transformer jako verifier.

Instalacja w HA:

1. Dodaj repozytorium jako add-on repository.
2. Zainstaluj add-on `nanoWakeWord`.
3. Wgraj modele `.onnx` do `/share/nanowakeword`.
4. Uruchom add-on.
5. Wybierz usługę Wyoming w Voice Assist.

Docker na osobnym komputerze:

```bash
sudo mkdir -p /opt/wyoming-nanowakeword/{data,share/nanowakeword}
sudo cp /path/to/*.onnx /opt/wyoming-nanowakeword/share/nanowakeword/
docker compose -f compose.yml up -d
```

W Home Assistant dodaj integrację Wyoming z adresem IP hosta Docker i portem
`10400`.

## References

- `rhasspy/wyoming-openwakeword`
- `arcosoph/nanowakeword`
- Home Assistant Wyoming integration: https://www.home-assistant.io/integrations/wyoming/
- Home Assistant wake words: https://www.home-assistant.io/voice_control/create_wake_word/
