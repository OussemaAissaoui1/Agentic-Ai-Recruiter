/**
 * HeadTTSLipSync — minimum-latency streaming TTS + Oculus visemes.
 *
 *   SSE token  ──►  feedText()  ──►  per-sentence synthesize()
 *                                         │
 *                                         ▼  (HeadTTS worker, Kokoro)
 *                                  audio chunk + viseme timing
 *                                         │
 *                                         ▼
 *                       _scheduleChunk()   src.start(scheduledStart)
 *                                          scheduledStart = max(now+ε, chainTail)
 *                                          chainTail      = scheduledStart + duration
 *
 *   Web Audio's sample-accurate scheduling stitches consecutive chunks
 *   into one continuous stream — there is *no* gap as long as synth
 *   produces chunk N+1 before chunk N's playback ends.  With WebGPU
 *   (~3× realtime) Kokoro stays well ahead of playback, so a paragraph
 *   plays as one unbroken audio stream with phoneme-accurate viseme
 *   timing throughout.
 *
 *   Latency budget (TTFA = time to first audio):
 *     LLM_first_sentence + worker_synth(first_sentence) + ε(≈10 ms)
 *
 *   No pre-buffer is added.  We start playback the instant the first
 *   chunk lands, which is the theoretical minimum for sentence-level
 *   streaming.  If chain-tail ever falls behind real-time on a slow
 *   backend (WASM with no WebGPU), we re-anchor to now+ε — that's a
 *   hardware bottleneck (synth is slower than playback), not a JS bug.
 */

import { HeadTTS } from "@met4citizen/headtts";

export const VISEME_NAMES = [
  "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
  "viseme_kk",  "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
  "viseme_aa",  "viseme_E",  "viseme_I",  "viseme_O",  "viseme_U",
];

const SENTENCE_BOUNDARY  = /([.!?])(\s+|$)/;
const CROSSFADE_SEC      = 0.085;
const TARGET_WEIGHT_CAP  = 0.70;
const DEFAULT_LERP       = 0.22;
const ACTIVITY_LINGER_MS = 220;
const SCHEDULE_EPSILON   = 0.012;   // 12 ms — minimum lookahead for src.start

export class HeadTTSLipSync {
  constructor({ voice = "af_heart", language = "en-us", speed = 1.0 } = {}) {
    this.voice    = voice;
    this.language = language;
    this.speed    = speed;

    this.tts          = null;
    this.ready        = false;
    this.connectingP  = null;

    // Live AudioContext-scheduled segments. Each:
    //   { src, startTime, endTime, timeline:[{name, absTime, duration}] }
    this.segments     = [];
    this._chainTail   = 0;            // ctx time of next free slot
    this._inFlight    = 0;
    this._burstSeq    = 0;            // increments on each clear() / new burst
    this._activeBurst = 0;

    this._textBuffer        = "";
    this._lastActivity      = false;
    this._activityLingerT   = null;
    this.onActivityChange   = null;
    this.onError            = null;
    this.onLoadProgress     = null;

    this.currentWeights = {};
    for (const n of VISEME_NAMES) this.currentWeights[n] = 0;
    this.currentWeights["viseme_sil"] = 1;
  }

  // ────────────── connection ──────────────

  async init() {
    if (this.ready) return this;
    if (this.connectingP) return this.connectingP;

    this.connectingP = (async () => {
      const base       = window.location.origin;
      const workerURL  = `${base}/modules/worker-tts.mjs`;
      const dictURL    = `${base}/dictionaries/`;

      const tts = new HeadTTS({
        endpoints:        ["webgpu", "wasm"],
        workerModule:     workerURL,
        dictionaryURL:    dictURL,
        voiceURL:
          "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices",
        languages:        [this.language],
        voices:           [this.voice],
        defaultVoice:     this.voice,
        defaultLanguage:  this.language,
        defaultSpeed:     this.speed,
        defaultAudioEncoding: "wav",
        splitSentences:   true,    // HeadTTS may sub-split inside a single sentence
        splitLength:      1500,
      });

      tts.onerror = (err) => {
        console.error("[HeadTTS] error:", err);
        this.onError?.(err);
      };

      try {
        await tts.connect(null, (e) => {
          if (e?.loaded && e?.total) {
            this.onLoadProgress?.(Math.round((e.loaded / e.total) * 100));
          }
        });
        tts.setup({ voice: this.voice, language: this.language, speed: this.speed });
        this.tts   = tts;
        this.ready = true;
        this.onLoadProgress?.(100);
        console.log("[HeadTTS] ready");
      } catch (e) {
        console.error("[HeadTTS] connect failed:", e);
        this.onError?.(e);
        throw e;
      }
    })();

    return this.connectingP.then(() => this);
  }

  // ────────────── streaming text in ──────────────

  /** Begin a new burst — call when the user sends a new message. */
  beginBurst() {
    this.clear();
    this._activeBurst = ++this._burstSeq;
  }

  /** Append a streamed token; fires synth on every sentence boundary. */
  feedText(text) {
    if (!text) return;
    this._textBuffer += text;
    if (!this.ready) return;
    if (!this._activeBurst) this._activeBurst = ++this._burstSeq;

    while (true) {
      const m = this._textBuffer.match(SENTENCE_BOUNDARY);
      if (!m) break;
      const cut      = m.index + m[0].length;
      const sentence = this._textBuffer.slice(0, cut).trim();
      this._textBuffer = this._textBuffer.slice(cut);
      if (sentence) this._synthSentence(sentence);
    }
  }

  /** SSE `done` — synthesise any partial sentence still in the buffer. */
  flushFinal() {
    const tail = this._textBuffer.trim();
    this._textBuffer = "";
    if (!tail) return;
    if (!this.ready) {
      console.warn("[HeadTTS] flushFinal: not ready, dropped", tail.length, "chars");
      return;
    }
    if (!this._activeBurst) this._activeBurst = ++this._burstSeq;
    this._synthSentence(tail);
  }

  _synthSentence(text) {
    const burst = this._activeBurst;
    this._inFlight += 1;

    // Per-call onmessage fires for every audio chunk produced for THIS
    // synth request, BEFORE the global `tts.onmessage` (HeadTTS suppresses
    // the global handler when a per-call one is present).
    this.tts
      .synthesize({ input: text }, (msg) => {
        if (burst !== this._activeBurst) return;            // user moved on
        if (msg && msg.type === "audio" && msg.data?.audio) {
          this._scheduleChunk(msg.data);
        }
      })
      .catch((e) => console.warn("[HeadTTS] synth failed:", e))
      .finally(() => {
        this._inFlight = Math.max(0, this._inFlight - 1);
      });
  }

  // ────────────── chained AudioContext scheduling ──────────────

  _scheduleChunk(data) {
    const ctx = this.tts?.settings?.audioCtx;
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});

    const buffer = data.audio;
    const now    = ctx.currentTime;

    // Sample-accurate chain: if we still have a future tail, butt the new
    // chunk right against it (zero-gap continuation).  If the chain has
    // already drained past `now`, anchor at now+ε.
    const startTime = (this._chainTail > now) ? this._chainTail : (now + SCHEDULE_EPSILON);
    const endTime   = startTime + buffer.duration;
    this._chainTail = endTime;

    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    try {
      src.start(startTime);
    } catch (e) {
      console.warn("[HeadTTS] start failed:", e);
      return;
    }

    // Build viseme timeline at absolute AudioContext times.
    const visemes    = data.visemes    || [];
    const vtimes     = data.vtimes     || [];
    const vdurations = data.vdurations || [];
    const timeline   = new Array(visemes.length);
    for (let i = 0; i < visemes.length; i++) {
      timeline[i] = {
        name:     `viseme_${visemes[i]}`,
        absTime:  startTime + (vtimes[i] || 0) / 1000,
        duration: (vdurations[i] || 60) / 1000,
      };
    }

    const seg = { src, startTime, endTime, timeline };
    this.segments.push(seg);
    this._setActivity(true);

    src.onended = () => {
      const i = this.segments.indexOf(seg);
      if (i !== -1) this.segments.splice(i, 1);
      if (this.segments.length === 0 && this._inFlight === 0) {
        this._chainTail = 0;
        this._activeBurst = 0;
        this._setActivity(false);
      }
    };
  }

  // ────────────── activity flag (with small linger) ──────────────

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

  // ────────────── per-frame viseme weights ──────────────

  getVisemeWeights(lerpFactor = DEFAULT_LERP) {
    const target = {};
    for (const n of VISEME_NAMES) target[n] = 0;

    const ctx = this.tts?.settings?.audioCtx;
    if (ctx && this.segments.length > 0) {
      const now = ctx.currentTime;
      // Find the segment whose [startTime, endTime) contains `now`.
      // Segments are appended in scheduling order so a linear scan is fine.
      let active = null;
      for (const s of this.segments) {
        if (s.startTime <= now && now < s.endTime) { active = s; break; }
      }
      if (active) {
        for (const v of active.timeline) {
          const start = v.absTime;
          const end   = start + v.duration;
          const fade  = Math.min(CROSSFADE_SEC, Math.max(0.04, v.duration * 0.5));
          if (now < start - fade || now > end + fade) continue;

          let w;
          if (now < start)      w = 1 - (start - now) / fade;
          else if (now < end)   w = 1;
          else                  w = 1 - (now - end) / fade;
          w = Math.max(0, Math.min(1, w)) * TARGET_WEIGHT_CAP;

          if (target[v.name] === undefined) continue;
          if (w > target[v.name]) target[v.name] = w;
        }
      } else {
        target["viseme_sil"] = 1;
      }
    } else {
      target["viseme_sil"] = 1;
    }

    for (const n of VISEME_NAMES) {
      this.currentWeights[n] += (target[n] - this.currentWeights[n]) * lerpFactor;
      if (this.currentWeights[n] < 0.0015) this.currentWeights[n] = 0;
    }
    return this.currentWeights;
  }

  // ────────────── misc ──────────────

  async resumeAudio() {
    const ctx = this.tts?.settings?.audioCtx;
    if (ctx && ctx.state === "suspended") {
      try { await ctx.resume(); } catch {}
    }
  }

  /** Stop everything and drop pending. New synth output for prior burst is ignored. */
  clear() {
    this._textBuffer  = "";
    this._activeBurst = 0;            // any in-flight chunks become stale
    this._chainTail   = 0;
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
