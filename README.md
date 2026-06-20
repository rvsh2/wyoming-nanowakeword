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
  hey_home:
    phrase: "Hey Home"
    language: "en"
    architecture: "bcresnet"
    version: "v1"
```

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
- `debug_logging`: verbose logs

## Docker Compose on Another Computer

Create the directories and copy `.onnx` models:

```bash
sudo mkdir -p /opt/wyoming-nanowakeword/{data,share/nanowakeword}
sudo cp /path/to/*.onnx /opt/wyoming-nanowakeword/share/nanowakeword/
```

Start the service:

```bash
docker compose -f compose.yml up -d
```

The server listens on `0.0.0.0:10400`. In Home Assistant, add a Wyoming
integration pointing at the Docker host IP and port `10400`.

For local development builds:

```bash
docker compose -f compose.yml -f compose.override.example.yml up --build
```

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
- `bcresnet` jest rekomendowaną domyślną architekturą dla nowych modeli.

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
