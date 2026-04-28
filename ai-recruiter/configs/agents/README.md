# configs/agents/

Per-agent overlays — empty for now. Today every agent reads its
configuration from the top-level `configs/runtime.yaml`. As agents grow
config surface area, drop a `<name>.yaml` here and load it in the
agent's `startup()` hook.
