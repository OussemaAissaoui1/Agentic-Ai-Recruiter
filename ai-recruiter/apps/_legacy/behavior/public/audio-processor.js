// AudioWorkletProcessor for capturing microphone audio and buffering it
// before posting to the main thread for WebSocket transmission.
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._sampleCount = 0;
    this._threshold = 8000; // ~0.5s at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, mono channel 0
    this._buffer.push(new Float32Array(samples));
    this._sampleCount += samples.length;

    if (this._sampleCount >= this._threshold) {
      const merged = new Float32Array(this._sampleCount);
      let offset = 0;
      for (const chunk of this._buffer) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      this._buffer = [];
      this._sampleCount = 0;

      this.port.postMessage({ type: 'audio', samples: merged });
    }

    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
