# GPU budget

The unified app runs multiple ML pipelines concurrently. Default budget
assumes a single GPU; CPU fallback is wired everywhere small enough to
matter.

## Default split (24 GB GPU, e.g. 4090 / L40)

| Component | Device | Memory budget |
|---|---|---|
| vLLM Llama-3.1-8B (`AIR_GPU_VLLM_FRACTION=0.55`) | GPU | ~13 GB |
| HSEmotion EfficientNet-B0 | GPU | ~50 MB |
| Wav2Vec2-base | GPU | ~360 MB |
| MiniLM-L6-v2 + TAPJFNN + GNN | GPU | ~400 MB |
| MediaPipe FaceMesh / Pose / Hands | CPU | n/a |
| Qwen2.5-1.5B scorer + refiner | CPU | ~6 GB RAM |
| Kokoro TTS | CPU | ~200 MB RAM |

Headroom: ~9 GB free GPU. Adjust `AIR_GPU_VLLM_FRACTION` if you need to
free memory for a second concurrent vision agent or larger Wav2Vec2.

## 16 GB GPU (e.g. 4060 Ti, T4)

Set:
```bash
export AIR_GPU_VLLM_FRACTION=0.45
```
That gives vLLM ~7 GB and leaves ~8 GB for everything else. Llama-3.1-8B
fits in ~7 GB at bf16 with paged attention — confirmed.

If you can't fit vLLM at all, run NLP without GPU by setting
`enable_vllm: false` in `configs/agents/nlp.yaml` (planned — today the
agent will stay in `loading` indefinitely if the GPU is too small).

## Multi-GPU

vLLM supports tensor parallelism via `tensor_parallel_size`. The current
agent passes `tensor_parallel_size=0` (auto). To pin to one specific
device, set `CUDA_VISIBLE_DEVICES` before starting uvicorn.

## What competes for the GPU

The vision agent's StudentNet, visual MLP and HSEmotion all try GPU
first; they're tiny and rarely a real cost. The matching agent's GNN
encodes 14 nodes — also tiny. The serious tenant is vLLM. Keep its
fraction conservative if you start hitting OOM on Wav2Vec2 calls during
WebSocket streaming.
