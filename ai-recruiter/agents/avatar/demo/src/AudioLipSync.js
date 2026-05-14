/**
 * AudioLipSync — Real-time audio analysis for avatar lip synchronization.
 *
 * Provides two lip-sync strategies:
 *   1. Server-side visemes: Pre-computed viseme timelines sent via SSE
 *   2. Client-side analysis: Real-time FFT frequency-band → viseme mapping
 *
 * Both can run simultaneously for best results: server visemes provide
 * accurate phoneme timing, client analysis fills gaps and provides
 * smooth transitions.
 *
 * Ready Player Me viseme blend shape targets:
 *   visemeSil, visemePP, visemeFF, visemeTH, visemeDD,
 *   visemeK, visemeCH, visemeSS, visemeNN, visemeRR,
 *   visemeAA, visemeE, visemeI, visemeO, visemeU, visemeOH
 */

// Ready Player Me (Wolf3D) morph-target names
const VISEME_NAMES = [
  "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
  "viseme_kk",  "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
  "viseme_aa",  "viseme_E",  "viseme_I",  "viseme_O",  "viseme_U",
];

/**
 * Audio queue that plays WAV chunks sequentially and exposes
 * an AnalyserNode for real-time frequency analysis.
 */
export class LipSyncAudioQueue {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.queue = [];
    this.playing = false;
    this.onActivityChange = null;

    // Server-side viseme schedule
    this._visemeSchedule = [];
    this._currentChunkStart = 0;

    // Client-side analysis buffer
    this._freqData = null;
    this._timeData = null;

    // Current blended viseme weights
    this.currentWeights = {};
    VISEME_NAMES.forEach((n) => (this.currentWeights[n] = 0));
  }

  _getCtx() {
    if (!this.ctx || this.ctx.state === "closed") {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.6;
      this._freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this._timeData = new Uint8Array(this.analyser.fftSize);
    }
    if (this.ctx.state === "suspended") {
      this.ctx.resume();
    }
    return this.ctx;
  }

  /**
   * Enqueue a base64-encoded WAV chunk for playback.
   * @param {string} base64Wav - Base64-encoded WAV audio data
   * @param {Array|null} visemes - Optional server-side viseme timeline for this chunk
   */
  async enqueue(base64Wav, visemes = null) {
    const ctx = this._getCtx();
    const binary = atob(base64Wav);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    try {
      const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
      this.queue.push({ buffer: audioBuffer, visemes });
      if (!this.playing) this._playNext();
    } catch (e) {
      console.error("[LipSyncAudio] decode error:", e);
    }
  }

  _playNext() {
    if (this.queue.length === 0) {
      this.playing = false;
      this._visemeSchedule = [];
      this.onActivityChange?.(false);
      return;
    }

    this.playing = true;
    this.onActivityChange?.(true);

    const ctx = this._getCtx();
    const { buffer, visemes } = this.queue.shift();

    // Schedule visemes for this chunk
    if (visemes && visemes.length > 0) {
      this._visemeSchedule = visemes;
      this._currentChunkStart = ctx.currentTime;
    } else {
      this._visemeSchedule = [];
    }

    const src = ctx.createBufferSource();
    src.buffer = buffer;

    // Route through analyser for real-time frequency data
    src.connect(this.analyser);
    this.analyser.connect(ctx.destination);

    src.onended = () => this._playNext();
    src.start();
  }

  /**
   * Get current viseme weights. Call this every animation frame.
   * Blends server-side viseme data with client-side frequency analysis.
   *
   * @param {number} lerpFactor - Smoothing factor (0-1, lower = smoother)
   * @returns {Object} Map of viseme name → weight (0-1)
   */
  getVisemeWeights(lerpFactor = 0.35) {
    if (!this.playing || !this.analyser) {
      // Decay to zero when not playing
      for (const name of VISEME_NAMES) {
        this.currentWeights[name] *= 0.92;
        if (this.currentWeights[name] < 0.01) this.currentWeights[name] = 0;
      }
      return this.currentWeights;
    }

    // Get real-time frequency data from analyser
    this.analyser.getByteFrequencyData(this._freqData);
    this.analyser.getByteTimeDomainData(this._timeData);

    const clientWeights = this._analyzeFrequencyData();

    // Check for server-side visemes
    let serverWeights = null;
    if (this._visemeSchedule.length > 0 && this.ctx) {
      const elapsed = this.ctx.currentTime - this._currentChunkStart;
      serverWeights = this._getServerVisemeAt(elapsed);
    }

    // Blend: prefer server visemes when available, fill with client analysis
    const targetWeights = {};
    for (const name of VISEME_NAMES) {
      const server = serverWeights?.[name] ?? 0;
      const client = clientWeights[name] ?? 0;

      if (serverWeights) {
        // 70% server, 30% client for smooth blending
        targetWeights[name] = server * 0.7 + client * 0.3;
      } else {
        targetWeights[name] = client;
      }
    }

    // Lerp for smooth transitions
    for (const name of VISEME_NAMES) {
      const target = targetWeights[name] ?? 0;
      this.currentWeights[name] +=
        (target - this.currentWeights[name]) * lerpFactor;
    }

    return this.currentWeights;
  }

  /**
   * Client-side frequency analysis → viseme weights.
   * Maps FFT frequency bands to articulatory features.
   */
  _analyzeFrequencyData() {
    const weights = {};
    VISEME_NAMES.forEach((n) => (weights[n] = 0));

    if (!this._freqData) return weights;

    const binCount = this._freqData.length;
    const sampleRate = this.ctx?.sampleRate || 48000;
    const binWidth = sampleRate / (binCount * 2);

    // Helper: average energy in a frequency range
    const bandEnergy = (lowHz, highHz) => {
      const lowBin = Math.max(0, Math.floor(lowHz / binWidth));
      const highBin = Math.min(binCount - 1, Math.floor(highHz / binWidth));
      if (highBin <= lowBin) return 0;
      let sum = 0;
      for (let i = lowBin; i <= highBin; i++) sum += this._freqData[i];
      return sum / (highBin - lowBin + 1) / 255;
    };

    // Extract band energies
    const voice = bandEnergy(80, 300);
    const f1 = bandEnergy(300, 1000);
    const f2 = bandEnergy(1000, 3000);
    const fric = bandEnergy(3000, 6000);
    const sib = bandEnergy(6000, 12000);

    // Overall energy (for gating)
    const overall =
      (voice * 0.3 + f1 * 0.3 + f2 * 0.2 + fric * 0.1 + sib * 0.1);

    if (overall < 0.03) {
      weights["viseme_sil"] = 1.0;
      return weights;
    }

    const energy = Math.min(1.0, overall * 3);

    // Vowels (RPM names)
    weights["viseme_aa"] = Math.min(1.0, f1 * 2.0 * energy);
    weights["viseme_E"]  = Math.min(1.0, f2 * 1.4 * energy * (1 - f1 * 0.4));
    weights["viseme_I"]  = Math.min(1.0, f2 * 1.6 * (1 - f1) * energy);
    weights["viseme_O"]  = Math.min(1.0, f1 * (1 - f2 * 0.7) * energy * 1.5);
    weights["viseme_U"]  = Math.min(1.0, (1 - f2) * (1 - f1 * 0.4) * energy * 0.9);

    // Fricatives
    weights["viseme_SS"] = Math.min(1.0, sib * 2.5 * energy);
    weights["viseme_FF"] = Math.min(1.0, fric * 1.8 * (1 - sib) * energy);
    weights["viseme_TH"] = Math.min(1.0, fric * 0.9 * energy);

    // Stops/nasals
    weights["viseme_PP"] = Math.min(1.0, energy * 0.7 * (1 - voice) * (1 - fric));
    weights["viseme_DD"] = Math.min(1.0, energy * 0.6 * voice * (1 - f1 * 0.4));
    weights["viseme_kk"] = Math.min(1.0, energy * 0.5 * (1 - voice * 0.5) * fric * 0.6);
    weights["viseme_CH"] = Math.min(1.0, fric * sib * energy * 1.5);
    weights["viseme_nn"] = Math.min(1.0, voice * 0.6 * (1 - fric) * energy * 0.7);
    weights["viseme_RR"] = Math.min(1.0, f1 * f2 * energy * 0.9);

    // Silence (inverse)
    weights["viseme_sil"] = Math.max(0, 1.0 - energy * 3);

    return weights;
  }

  /**
   * Look up server-side viseme weights at a given time offset.
   */
  _getServerVisemeAt(elapsed) {
    if (!this._visemeSchedule || this._visemeSchedule.length === 0) return null;

    // Binary search for the right frame
    let best = null;
    for (const frame of this._visemeSchedule) {
      if (frame.time <= elapsed && frame.time + frame.duration > elapsed) {
        best = frame.weights;
        break;
      }
      if (frame.time > elapsed) break;
    }

    return best;
  }

  clear() {
    this.queue = [];
    this.playing = false;
    this._visemeSchedule = [];
    this.onActivityChange?.(false);
  }

  destroy() {
    this.clear();
    if (this.ctx && this.ctx.state !== "closed") {
      this.ctx.close().catch(() => {});
    }
  }
}

export { VISEME_NAMES };
