/**
 * StreamingLipSync — server-driven, minimum-latency lip-sync.
 *
 *   FastAPI ─SSE─►  audio   (base64 WAV, one per sentence,
 *                            Kokoro on GPU @ ~150× realtime)
 *           ─SSE─►  visemes (per-frame Oculus 15 weight map)
 *                       │
 *                       ▼
 *                  enqueueAudio() / enqueueVisemes()
 *                       │
 *                       ▼
 *           sample-accurate Web-Audio chained playback
 *                       │
 *               ┌───────┴────────┐
 *               ▼                ▼
 *           speakers         viseme weights → GLB morph targets
 *
 * No in-browser TTS — synthesis runs on the server's GPU, so TTFA is
 * dominated by network RTT (~10–50 ms) and the LLM's first-sentence
 * generation, not by ONNX-on-WASM.
 *
 * Continuity: each chunk is scheduled at `max(_chainTail, now+ε)`. As
 * long as the server keeps the chain ahead of real-time (it does — Kokoro
 * is two orders of magnitude faster than playback on GPU), consecutive
 * chunks butt up against each other on the AudioContext timeline with
 * sample accuracy. Zero gap, zero per-chunk fade-out.
 */

export const VISEME_NAMES = [
  "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
  "viseme_kk",  "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
  "viseme_aa",  "viseme_E",  "viseme_I",  "viseme_O",  "viseme_U",
];

// Extra articulator morph targets we'll drive from the viseme table.
// All exist on the GLB (verified earlier).
const ARTICULATOR_NAMES = [
  "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
  "mouthSmileLeft", "mouthSmileRight",
];

/**
 * HeadTTS / TalkingHead-style articulation table — each viseme drives
 * a specific blendshape profile rather than a single morph weight.
 * Values follow Mika Suominen's TalkingHead conventions: closures
 * (PP, FF) explicitly close the lips, rounded vowels (O, U) engage
 * funnel + pucker, front vowels (E, I) engage smile, etc.
 *
 *  jaw     — jawOpen
 *  close   — mouthClose (lips together)
 *  funnel  — mouthFunnel (round + forward)
 *  pucker  — mouthPucker (round + close)
 *  smile   — mouthSmile{Left,Right}
 *  intensity — strength applied to viseme_* morph itself
 */
// Calibrated for natural conversational speech.  Closures (PP/FF/M/N) get
// real lip closure, vowels get appropriate openings, rounded vowels round —
// but no setting is dialed to "stage-actor" extremes.
const VISEME_ARTICULATION = {
  viseme_sil: { jaw: 0.00, close: 0.04, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.00 },
  viseme_PP:  { jaw: 0.00, close: 0.70, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.80 }, // P, B, M
  viseme_FF:  { jaw: 0.04, close: 0.18, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.65 }, // F, V
  viseme_TH:  { jaw: 0.08, close: 0.00, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.55 }, // TH
  viseme_DD:  { jaw: 0.10, close: 0.10, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.55 }, // D, T
  viseme_kk:  { jaw: 0.12, close: 0.00, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.55 }, // K, G
  viseme_CH:  { jaw: 0.08, close: 0.00, funnel: 0.32, pucker: 0.22, smile: 0.00, intensity: 0.65 }, // CH, SH
  viseme_SS:  { jaw: 0.04, close: 0.18, funnel: 0.00, pucker: 0.00, smile: 0.04, intensity: 0.60 }, // S, Z
  viseme_nn:  { jaw: 0.08, close: 0.50, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.65 }, // N, NG
  viseme_RR:  { jaw: 0.14, close: 0.00, funnel: 0.16, pucker: 0.08, smile: 0.00, intensity: 0.70 }, // R
  viseme_aa:  { jaw: 0.42, close: 0.00, funnel: 0.00, pucker: 0.00, smile: 0.00, intensity: 0.85 }, // AH (open)
  viseme_E:   { jaw: 0.24, close: 0.00, funnel: 0.00, pucker: 0.00, smile: 0.12, intensity: 0.78 }, // EH
  viseme_I:   { jaw: 0.10, close: 0.00, funnel: 0.00, pucker: 0.00, smile: 0.18, intensity: 0.70 }, // IH
  viseme_O:   { jaw: 0.32, close: 0.00, funnel: 0.40, pucker: 0.30, smile: 0.00, intensity: 0.85 }, // OH
  viseme_U:   { jaw: 0.16, close: 0.00, funnel: 0.50, pucker: 0.40, smile: 0.00, intensity: 0.78 }, // OO
};

const ALL_TARGET_NAMES = [...VISEME_NAMES, ...ARTICULATOR_NAMES];

const SCHEDULE_EPSILON   = 0.012;  // seconds — minimum lookahead for src.start
const TARGET_WEIGHT_CAP  = 0.78;   // overall amplitude ceiling on viseme_* morphs
const DEFAULT_LERP       = 0.20;   // chase rate inside lipSync (slower = more natural)
const SILENCE_THRESHOLD  = 0.12;   // total dominant energy below this → mouth at rest
const ACTIVITY_LINGER_MS = 200;

export class StreamingLipSync {
  constructor() {
    this.audioCtx     = null;
    this.ready        = false;

    // Live AudioContext-scheduled segments.
    //   { src, startTime, endTime, frames:[{absTime, duration, weights}] }
    this.segments     = [];
    this._chainTail   = 0;
    this._inFlight    = 0;       // base64 → AudioBuffer decodes outstanding
    this._burstSeq    = 0;
    this._activeBurst = 0;

    // The server emits each chunk as TWO SSE events: `audio` first,
    // then `visemes`. We schedule the audio immediately (don't wait for
    // visemes), then patch the segment's viseme timeline in place when
    // the matching `visemes` event lands a few ms later.  Both then
    // share the same `startTime`, so they're sample-accurately aligned.
    this._pendingForFrames = [];   // segments awaiting their viseme frames
    this._unmatchedVisemes = [];   // visemes that arrived before audio (rare)

    this._lastActivity      = false;
    this._activityLingerT   = null;
    this.onActivityChange   = null;
    this.onError            = null;

    this.currentWeights = {};
    for (const n of ALL_TARGET_NAMES) this.currentWeights[n] = 0;
  }

  async init() {
    if (this.ready) return this;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = new Ctx({ latencyHint: "interactive" });
    this.ready = true;
    console.log("[StreamingLipSync] ready (sampleRate=" + this.audioCtx.sampleRate + ")");
    return this;
  }

  /** Mark a fresh "burst" (one user→avatar exchange). Drops in-flight stale audio. */
  beginBurst() {
    this.clear();
    this._activeBurst = ++this._burstSeq;
  }

  // No-op text APIs — synthesis happens on the server.
  feedText(_text) { /* unused */ }
  flushFinal()    { /* unused */ }

  // ────────────────────── SSE ingest ──────────────────────

  /**
   * Server emits a `visemes` event with a list of frames:
   *   [{ time, duration, weights: { viseme_aa: 0.2, ...} }, ...]
   * times are seconds relative to the audio chunk's start.
   *
   * Visemes always arrive AFTER the matching audio (server order). We
   * pop the oldest segment that's still waiting for frames and patch
   * its timeline in place using its already-known startTime.
   */
  enqueueVisemes(frames) {
    if (!Array.isArray(frames) || frames.length === 0) return;
    const seg = this._pendingForFrames.shift();
    if (seg) {
      seg.frames = this._toAbsFrames(frames, seg.startTime);
    } else {
      // Out-of-order: visemes arrived before audio. Stash them.
      this._unmatchedVisemes.push(frames);
    }
  }

  /**
   * Server emits an `audio` event with a base64-encoded WAV per sentence.
   * We schedule playback immediately, then attach the matching viseme
   * timeline when its event arrives.
   */
  async enqueueAudio(base64Wav) {
    if (!this.ready) await this.init();
    if (this._activeBurst === 0) this._activeBurst = ++this._burstSeq;

    const burst = this._activeBurst;

    this._inFlight += 1;
    try {
      const ctx = this.audioCtx;
      if (ctx.state === "suspended") ctx.resume().catch(() => {});

      // base64 → ArrayBuffer → AudioBuffer (decode is async)
      const binary = atob(base64Wav);
      const bytes  = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const buffer = await ctx.decodeAudioData(bytes.buffer);

      // If the user moved on while we were decoding, drop this chunk.
      if (burst !== this._activeBurst) return;

      // If visemes arrived before audio (rare), pair them up now.
      const earlyFrames = this._unmatchedVisemes.shift() || null;
      const seg = this._scheduleChunk(buffer, earlyFrames);
      if (seg && !earlyFrames) {
        this._pendingForFrames.push(seg);
      }
    } catch (e) {
      console.warn("[StreamingLipSync] enqueueAudio failed:", e);
      this.onError?.(e);
    } finally {
      this._inFlight = Math.max(0, this._inFlight - 1);
    }
  }

  /** Convert server frames to absolute-AC-time entries against a known start. */
  _toAbsFrames(frames, startTime) {
    const out = new Array(frames.length);
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      out[i] = {
        absTime:  startTime + (f.time || 0),
        duration: (f.duration || 0.02),
        weights:  f.weights || null,
      };
    }
    return out;
  }

  // ────────────────────── chained scheduling ──────────────────────

  _scheduleChunk(buffer, frames) {
    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    // Sample-accurate chain. If the chain still has a future tail, butt
    // the new chunk directly against it (zero-gap continuation).
    const startTime = (this._chainTail > now) ? this._chainTail : (now + SCHEDULE_EPSILON);
    const endTime   = startTime + buffer.duration;
    this._chainTail = endTime;

    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    try { src.start(startTime); }
    catch (e) { console.warn("[StreamingLipSync] start failed:", e); return null; }

    const absFrames = (frames && frames.length)
      ? this._toAbsFrames(frames, startTime)
      : [];

    const seg = { src, startTime, endTime, frames: absFrames };
    this.segments.push(seg);
    this._setActivity(true);

    src.onended = () => {
      const i = this.segments.indexOf(seg);
      if (i !== -1) this.segments.splice(i, 1);
      // Remove from waiting list if visemes never arrived
      const j = this._pendingForFrames.indexOf(seg);
      if (j !== -1) this._pendingForFrames.splice(j, 1);
      if (this.segments.length === 0 && this._inFlight === 0) {
        this._chainTail = 0;
        this._activeBurst = 0;
        this._setActivity(false);
      }
    };
    return seg;
  }

  // ────────────────────── per-frame viseme query ──────────────────────

  /**
   * Per-frame morph target weights — viseme_* + articulators (jawOpen,
   * mouthClose, mouthFunnel, mouthPucker, mouthSmile{Left,Right}).
   *
   * Algorithm:
   *   1. Find the top-2 visemes by weight in the current frame
   *   2. If their combined weight < SILENCE_THRESHOLD → silent (all zeros)
   *   3. Otherwise normalise the two weights to a ratio (r1 + r2 = 1)
   *      and linearly blend their VISEME_ARTICULATION profiles
   *   4. Drive only the top-1 + top-2 viseme_* (ratio-scaled), and the
   *      articulator targets from the blended profile
   *   5. Lerp current → target at lerpFactor
   *
   * The "pick top-2 + blend articulation profiles" is the same approach
   * met4citizen's TalkingHead uses to translate phoneme/viseme streams
   * into natural mouth motion. With this in place, closures (PP/FF/MM)
   * actually close the jaw/lips, rounded vowels round, etc.
   */
  getVisemeWeights(lerpFactor = DEFAULT_LERP) {
    const target = {};
    for (const n of ALL_TARGET_NAMES) target[n] = 0;

    const ctx = this.audioCtx;
    let activeFrame = null;
    if (ctx && this.segments.length > 0) {
      const now = ctx.currentTime;
      for (const seg of this.segments) {
        if (seg.startTime <= now && now < seg.endTime) {
          for (const f of seg.frames) {
            if (f.absTime <= now && now < f.absTime + f.duration) {
              activeFrame = f;
              break;
            }
          }
          break;
        }
      }
    }

    if (activeFrame && activeFrame.weights) {
      // Top-2 viseme by weight (skip the silence one — it's an absence-of-energy marker, not an articulation we want to mix in).
      let top1 = null, top2 = null;
      for (const [name, w] of Object.entries(activeFrame.weights)) {
        if (!VISEME_ARTICULATION[name]) continue;
        if (name === "viseme_sil") continue;
        if (!top1 || w > top1.w) { top2 = top1; top1 = { name, w }; }
        else if (!top2 || w > top2.w) { top2 = { name, w }; }
      }

      const totalEnergy = (top1?.w || 0) + (top2?.w || 0);
      if (top1 && totalEnergy >= SILENCE_THRESHOLD) {
        // Normalise into a ratio
        const sum = totalEnergy || 1;
        const r1  = (top1?.w || 0) / sum;
        const r2  = (top2?.w || 0) / sum;
        const a1  = VISEME_ARTICULATION[top1.name];
        const a2  = top2 ? VISEME_ARTICULATION[top2.name] : a1;

        // Articulators — linearly blended profile
        target.jawOpen          = r1 * a1.jaw    + r2 * a2.jaw;
        target.mouthClose       = r1 * a1.close  + r2 * a2.close;
        target.mouthFunnel      = r1 * a1.funnel + r2 * a2.funnel;
        target.mouthPucker      = r1 * a1.pucker + r2 * a2.pucker;
        target.mouthSmileLeft   = r1 * a1.smile  + r2 * a2.smile;
        target.mouthSmileRight  = r1 * a1.smile  + r2 * a2.smile;

        // viseme_* morphs — only the top two carry weight; intensity comes
        // from the articulation table so closures don't get fully driven.
        target[top1.name] = a1.intensity * TARGET_WEIGHT_CAP * r1;
        if (top2 && top2.name !== top1.name) {
          target[top2.name] = a2.intensity * TARGET_WEIGHT_CAP * r2;
        } else if (top2 && top2.name === top1.name) {
          target[top1.name] += a2.intensity * TARGET_WEIGHT_CAP * r2;
        }
      }
      // else: silence frame — leave all targets at 0 (mouth at GLB neutral)
    }
    // else: no active segment — mouth at rest (all zeros)

    for (const n of ALL_TARGET_NAMES) {
      const cur = this.currentWeights[n] || 0;
      const next = cur + (target[n] - cur) * lerpFactor;
      this.currentWeights[n] = next < 0.0015 ? 0 : next;
    }
    return this.currentWeights;
  }

  // ────────────────────── activity (with linger) ──────────────────────

  _setActivity(on) {
    if (on) {
      if (this._activityLingerT) {
        clearTimeout(this._activityLingerT);
        this._activityLingerT = null;
      }
      if (!this._lastActivity) {
        this._lastActivity = true;
        this.onActivityChange?.(true);
      }
    } else {
      if (this._activityLingerT) clearTimeout(this._activityLingerT);
      this._activityLingerT = setTimeout(() => {
        this._activityLingerT = null;
        if (this.segments.length === 0 && this._inFlight === 0) {
          this._lastActivity = false;
          this.onActivityChange?.(false);
        }
      }, ACTIVITY_LINGER_MS);
    }
  }

  // ────────────────────── misc ──────────────────────

  async resumeAudio() {
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      try { await this.audioCtx.resume(); } catch {}
    }
  }

  clear() {
    this._activeBurst       = 0;
    this._chainTail         = 0;
    this._pendingForFrames  = [];
    this._unmatchedVisemes  = [];
    for (const s of this.segments) {
      try { s.src.onended = null; s.src.stop(); } catch {}
    }
    this.segments = [];
    if (this._activityLingerT) {
      clearTimeout(this._activityLingerT);
      this._activityLingerT = null;
    }
    if (this._lastActivity) {
      this._lastActivity = false;
      this.onActivityChange?.(false);
    }
  }

  destroy() { this.clear(); }
}
