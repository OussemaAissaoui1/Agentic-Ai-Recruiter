/**
 * AudioLipSync — real-time audio analysis for avatar lip-sync.
 *
 * Server-side visemes (when present) are blended with client-side FFT
 * frequency-band → viseme mapping. Both paths live behind a single
 * `getVisemeWeights()` consumed by the avatar's animation loop.
 *
 * Ported from /apps/_legacy/demo/src/AudioLipSync.js. Public surface:
 *   new LipSyncAudioQueue()
 *   .enqueue(base64Wav: string, visemes?: VisemeFrame[])
 *   .getVisemeWeights(lerpFactor?: number): Record<VisemeName, number>
 *   .clear() / .destroy()
 *   .onActivityChange?: (playing: boolean) => void
 */

export const VISEME_NAMES = [
  "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
  "viseme_kk",  "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
  "viseme_aa",  "viseme_E",  "viseme_I",  "viseme_O",  "viseme_U",
] as const;

export type VisemeName = (typeof VISEME_NAMES)[number];
export type VisemeWeights = Record<VisemeName, number>;

export interface VisemeFrame {
  time: number;       // seconds from start of chunk
  duration: number;   // seconds
  weights: Partial<VisemeWeights>;
}

interface QueueItem {
  buffer: AudioBuffer;
  visemes: VisemeFrame[] | null;
}

export class LipSyncAudioQueue {
  ctx: AudioContext | null = null;
  analyser: AnalyserNode | null = null;
  queue: QueueItem[] = [];
  playing = false;
  onActivityChange?: (playing: boolean) => void;

  private _visemeSchedule: VisemeFrame[] = [];
  private _currentChunkStart = 0;
  private _freqData: Uint8Array<ArrayBuffer> | null = null;
  private _timeData: Uint8Array<ArrayBuffer> | null = null;

  currentWeights: VisemeWeights;

  constructor() {
    this.currentWeights = Object.fromEntries(
      VISEME_NAMES.map((n) => [n, 0]),
    ) as VisemeWeights;
  }

  private _getCtx(): AudioContext {
    if (!this.ctx || this.ctx.state === "closed") {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      this.ctx = new Ctor();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.6;
      this._freqData = new Uint8Array(new ArrayBuffer(this.analyser.frequencyBinCount));
      this._timeData = new Uint8Array(new ArrayBuffer(this.analyser.fftSize));
    }
    if (this.ctx.state === "suspended") {
      void this.ctx.resume();
    }
    return this.ctx;
  }

  async enqueue(base64Wav: string, visemes: VisemeFrame[] | null = null): Promise<void> {
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

  private _playNext(): void {
    if (this.queue.length === 0) {
      this.playing = false;
      this._visemeSchedule = [];
      this.onActivityChange?.(false);
      return;
    }

    this.playing = true;
    this.onActivityChange?.(true);

    const ctx = this._getCtx();
    const item = this.queue.shift();
    if (!item) return;

    if (item.visemes && item.visemes.length > 0) {
      this._visemeSchedule = item.visemes;
      this._currentChunkStart = ctx.currentTime;
    } else {
      this._visemeSchedule = [];
    }

    const src = ctx.createBufferSource();
    src.buffer = item.buffer;
    if (this.analyser) {
      src.connect(this.analyser);
      this.analyser.connect(ctx.destination);
    } else {
      src.connect(ctx.destination);
    }
    src.onended = () => this._playNext();
    src.start();
  }

  getVisemeWeights(lerpFactor = 0.35): VisemeWeights {
    if (!this.playing || !this.analyser) {
      for (const name of VISEME_NAMES) {
        this.currentWeights[name] *= 0.92;
        if (this.currentWeights[name] < 0.01) this.currentWeights[name] = 0;
      }
      return this.currentWeights;
    }

    if (this._freqData) this.analyser.getByteFrequencyData(this._freqData);
    if (this._timeData) this.analyser.getByteTimeDomainData(this._timeData);

    const clientWeights = this._analyzeFrequencyData();

    let serverWeights: Partial<VisemeWeights> | null = null;
    if (this._visemeSchedule.length > 0 && this.ctx) {
      const elapsed = this.ctx.currentTime - this._currentChunkStart;
      serverWeights = this._getServerVisemeAt(elapsed);
    }

    const targetWeights: VisemeWeights = { ...this.currentWeights };
    for (const name of VISEME_NAMES) {
      const server = serverWeights?.[name] ?? 0;
      const client = clientWeights[name] ?? 0;
      targetWeights[name] = serverWeights ? server * 0.7 + client * 0.3 : client;
    }

    for (const name of VISEME_NAMES) {
      const target = targetWeights[name] ?? 0;
      this.currentWeights[name] += (target - this.currentWeights[name]) * lerpFactor;
    }

    return this.currentWeights;
  }

  private _analyzeFrequencyData(): VisemeWeights {
    const weights = Object.fromEntries(
      VISEME_NAMES.map((n) => [n, 0]),
    ) as VisemeWeights;

    if (!this._freqData) return weights;

    const binCount = this._freqData.length;
    const sampleRate = this.ctx?.sampleRate || 48000;
    const binWidth = sampleRate / (binCount * 2);

    const bandEnergy = (lowHz: number, highHz: number): number => {
      const lowBin = Math.max(0, Math.floor(lowHz / binWidth));
      const highBin = Math.min(binCount - 1, Math.floor(highHz / binWidth));
      if (highBin <= lowBin) return 0;
      let sum = 0;
      for (let i = lowBin; i <= highBin; i++) sum += this._freqData![i];
      return sum / (highBin - lowBin + 1) / 255;
    };

    const voice = bandEnergy(80, 300);
    const f1 = bandEnergy(300, 1000);
    const f2 = bandEnergy(1000, 3000);
    const fric = bandEnergy(3000, 6000);
    const sib = bandEnergy(6000, 12000);

    const overall = voice * 0.3 + f1 * 0.3 + f2 * 0.2 + fric * 0.1 + sib * 0.1;
    if (overall < 0.03) {
      weights["viseme_sil"] = 1.0;
      return weights;
    }
    const energy = Math.min(1.0, overall * 3);

    weights["viseme_aa"] = Math.min(0.75, f1 * 1.2 * energy);
    weights["viseme_E"]  = Math.min(0.65, f2 * 0.9 * energy * (1 - f1 * 0.4));
    weights["viseme_I"]  = Math.min(0.60, f2 * 1.0 * (1 - f1) * energy);
    weights["viseme_O"]  = Math.min(0.70, f1 * (1 - f2 * 0.7) * energy * 1.0);
    weights["viseme_U"]  = Math.min(0.55, (1 - f2) * (1 - f1 * 0.4) * energy * 0.7);

    weights["viseme_SS"] = Math.min(1.0, sib * 2.5 * energy);
    weights["viseme_FF"] = Math.min(1.0, fric * 1.8 * (1 - sib) * energy);
    weights["viseme_TH"] = Math.min(1.0, fric * 0.9 * energy);

    weights["viseme_PP"] = Math.min(1.0, energy * 0.7 * (1 - voice) * (1 - fric));
    weights["viseme_DD"] = Math.min(1.0, energy * 0.6 * voice * (1 - f1 * 0.4));
    weights["viseme_kk"] = Math.min(1.0, energy * 0.5 * (1 - voice * 0.5) * fric * 0.6);
    weights["viseme_CH"] = Math.min(1.0, fric * sib * energy * 1.5);
    weights["viseme_nn"] = Math.min(1.0, voice * 0.6 * (1 - fric) * energy * 0.7);
    weights["viseme_RR"] = Math.min(1.0, f1 * f2 * energy * 0.9);

    weights["viseme_sil"] = Math.max(0, 1.0 - energy * 3);
    return weights;
  }

  private _getServerVisemeAt(elapsed: number): Partial<VisemeWeights> | null {
    if (this._visemeSchedule.length === 0) return null;
    for (const frame of this._visemeSchedule) {
      if (frame.time <= elapsed && frame.time + frame.duration > elapsed) {
        return frame.weights;
      }
      if (frame.time > elapsed) break;
    }
    return null;
  }

  clear(): void {
    this.queue = [];
    this.playing = false;
    this._visemeSchedule = [];
    this.onActivityChange?.(false);
  }

  destroy(): void {
    this.clear();
    if (this.ctx && this.ctx.state !== "closed") {
      this.ctx.close().catch(() => {});
    }
  }
}
