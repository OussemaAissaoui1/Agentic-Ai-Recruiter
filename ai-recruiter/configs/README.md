# configs/

`runtime.yaml` is the single source of truth for the unified app's
runtime knobs. Loaded once via `configs.load_runtime()`; env vars
override hot fields (host, port, log level, GPU fraction).

Future per-agent overrides land under `configs/agents/<name>.yaml`.
