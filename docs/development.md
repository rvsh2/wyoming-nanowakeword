# Development

## Checks

```bash
python -m pytest
python -m ruff check .
python -m mypy wyoming_nanowakeword
docker compose -f compose.yml config
```

## Design Notes

The Wyoming wrapper converts Home Assistant audio to 16 kHz, 16-bit, mono PCM
and passes the resulting `np.ndarray[int16]` to `NanoInterpreter.predict`.

Do not add a second feature extraction pipeline in this repository. NanoWakeWord
owns streaming feature preparation and model architecture handling.

Architecture names in `models.yaml` are descriptive metadata only.
