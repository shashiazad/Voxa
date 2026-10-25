import {
  ConnectionState, DisconnectReason, RemoteTrack, Room, RoomEvent, Track, createLocalAudioTrack,
} from "livekit-client";
import type { CallSnapshot, Caption, Credentials, TurnMetrics, UiState } from "./types";

const SERVER_TO_UI: Record<string, UiState> = {
  IDLE: "listening", LISTENING: "listening", USER_SPEAKING: "listening", ENDPOINTING: "thinking",
  THINKING: "thinking", TOOL_CALL: "thinking", AGENT_SPEAKING: "speaking", INTERRUPT_PENDING: "paused",
  HANDOFF: "handoff", ENDED: "ended",
};
const dec = new TextDecoder();
const enc = new TextEncoder();

/** One voice call. Framework-agnostic: subscribe() receives an immutable snapshot on every change. */
export class VoxaCall {
  private room: Room | null = null;
  private listeners = new Set<(s: CallSnapshot) => void>();
  private snap: CallSnapshot = {
    state: "idle", serverState: "", muted: false, captions: [], metrics: [], agentLevel: 0, userLevel: 0,
  };
  private audioEls: HTMLMediaElement[] = [];
  private ctx: AudioContext | null = null;
  private agentAnalyser: AnalyserNode | null = null;
  private pendingGen: { gen: number; eventAt: number } | null = null;
  private lastSpeechEndAt = 0;
  private raf = 0;
  private creds: Credentials | null = null;
  private userEnded = false;
  private retries = 0;

  constructor(private opts: { maxRetries?: number } = {}) {}

  subscribe(fn: (s: CallSnapshot) => void): () => void {
    this.listeners.add(fn);
    fn(this.snap);
    return () => this.listeners.delete(fn);
  }
  get snapshot(): CallSnapshot { return this.snap; }
  private set(patch: Partial<CallSnapshot>) {
    this.snap = { ...this.snap, ...patch };
    this.listeners.forEach((l) => l(this.snap));
  }

  /** Ask for the microphone first so permission problems get a clear message instead of a dead call. */
  static async checkMic(): Promise<"granted" | "denied" | "unsupported" | "prompt"> {
    if (!navigator.mediaDevices?.getUserMedia) return "unsupported";
    try {
      const st = await (navigator.permissions?.query({ name: "microphone" as PermissionName }) ?? Promise.resolve(null));
      if (st && st.state === "denied") return "denied";
      if (st && st.state === "granted") return "granted";
    } catch { /* Safari/Firefox do not expose the permission name */ }
    return "prompt";
  }

  async start(creds: Credentials): Promise<void> {
    this.creds = creds;
    this.userEnded = false;
    this.set({ state: "connecting", error: undefined, captions: [], metrics: [], handoff: undefined });
    try {
      await this.connect(creds);
    } catch (e) {
      this.fail(e);
    }
  }

  private async connect(creds: Credentials) {
    // Created inside a user gesture (the Start click), so autoplay policies allow the agent's audio.
    this.ctx ??= new AudioContext({ latencyHint: "interactive" });
    await this.ctx.resume();
    const room = new Room({
      adaptiveStream: false,
      dynacast: false,
      stopLocalTrackOnUnpublish: true,
      audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      publishDefaults: { dtx: true, red: true, audioPreset: { maxBitrate: 32_000 } },
      disconnectOnPageLeave: true,
    });
    this.room = room;
    room
      .on(RoomEvent.TrackSubscribed, (t) => this.onTrack(t))
      .on(RoomEvent.DataReceived, (data, _p, _k, topic) => this.onData(topic ?? "", data))
      .on(RoomEvent.Reconnecting, () => { console.info("[voxa] reconnecting"); this.set({ state: "reconnecting" }); })
      .on(RoomEvent.Reconnected, () => { console.info("[voxa] reconnected"); this.set({ state: "listening" }); })
      .on(RoomEvent.Disconnected, (reason) => this.onDisconnected(reason))
      .on(RoomEvent.MediaDevicesError, (e) => this.fail(e));
    await room.connect(creds.url, creds.token, { autoSubscribe: true });
    const mic = await createLocalAudioTrack({ echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 });
    await room.localParticipant.publishTrack(mic, { source: Track.Source.Microphone, dtx: true });
    this.retries = 0;
    this.set({ state: "listening", muted: false });
    this.loop();
  }

  private onTrack(t: RemoteTrack) {
    if (t.kind !== Track.Kind.Audio || !this.ctx) return;
    const el = t.attach() as HTMLMediaElement;       // plays through the browser's AEC-aware output path
    el.style.display = "none";
    document.body.appendChild(el);
    this.audioEls.push(el);
    const src = this.ctx.createMediaStreamSource(new MediaStream([t.mediaStreamTrack]));
    this.agentAnalyser = this.ctx.createAnalyser();
    this.agentAnalyser.fftSize = 512;
    src.connect(this.agentAnalyser);
    // Chrome only pulls a remote WebRTC stream through the graph if it reaches the destination: use a muted gain.
    const mute = this.ctx.createGain(); mute.gain.value = 0;
    this.agentAnalyser.connect(mute); mute.connect(this.ctx.destination);
  }

  private level(an: AnalyserNode | null, buf: Uint8Array<ArrayBuffer>): number {
    if (!an) return 0;
    an.getByteTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) { const x = (v - 128) / 128; sum += x * x; }
    return Math.sqrt(sum / buf.length);
  }

  /** Per animation frame: output level for the UI, and the "first audible agent audio" acknowledgement. */
  private loop = () => {
    const buf = new Uint8Array(512) as Uint8Array<ArrayBuffer>;
    const tick = () => {
      if (!this.room) return;
      const lvl = this.level(this.agentAnalyser, buf);
      if (Math.abs(lvl - this.snap.agentLevel) > 0.02) this.set({ agentLevel: lvl });
      if (this.pendingGen && lvl > 0.01) {
        const { gen, eventAt } = this.pendingGen;
        const now = performance.now();
        this.pendingGen = null;
        this.send("voxa.ack", { gen, type: "first_audio_played", client_delay_ms: Math.round(now - eventAt) });
        if (this.lastSpeechEndAt) {
          // eos is detected ~160 ms after the user actually stopped; add it so the number is user-perceived
          const heard = Math.round(now - this.lastSpeechEndAt + 160);
          const metrics = this.snap.metrics.slice();
          const i = metrics.findIndex((m) => m.gen === gen);
          if (i >= 0) metrics[i] = { ...metrics[i], client_heard_ms: heard };
          else metrics.push({ gen, voice_to_voice_ms: null, stt_ms: null, llm_ttft_ms: null, tts_ttfb_ms: null, client_heard_ms: heard });
          this.set({ metrics });
        }
      }
      this.raf = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(tick);
  };

  private onData(topic: string, data: Uint8Array) {
    if (!topic.startsWith("voxa.")) return;
    let m: any = {};
    try { m = JSON.parse(dec.decode(data)); } catch { return; }
    switch (topic) {
      case "voxa.state": {
        const ui = SERVER_TO_UI[m.state];
        if (ui) this.set({ state: ui, serverState: m.state });
        break;
      }
      case "voxa.vad":
        if (m.event === "speech_end") this.lastSpeechEndAt = performance.now();
        break;
      case "voxa.transcript": {
        const caps = this.snap.captions.slice();
        const last = caps[caps.length - 1];
        const cap: Caption = { role: "user", text: m.text, final: !!m.final };
        if (last && last.role === "user" && !last.final) caps[caps.length - 1] = cap;
        else if (m.text) caps.push(cap);
        this.set({ captions: caps.slice(-40) });
        break;
      }
      case "voxa.agent_word": {   // word-synced captions: appended as each word is actually played
        const caps = this.snap.captions.slice();
        const last = caps[caps.length - 1];
        if (last && last.role === "agent" && last.gen === m.gen) caps[caps.length - 1] = { ...last, text: `${last.text} ${m.word}` };
        else caps.push({ role: "agent", text: m.word, final: false, gen: m.gen });
        this.set({ captions: caps.slice(-40) });
        break;
      }
      case "voxa.agent_audio_start":
        this.pendingGen = { gen: m.gen, eventAt: performance.now() };
        break;
      case "voxa.interrupt":
        if (m.stage === 2) {      // barge-in confirmed: the interrupted caption stays, marked as cut off
          const caps = this.snap.captions.map((c) => (c.role === "agent" && c.gen === m.gen ? { ...c, final: true, text: c.text + " …" } : c));
          this.set({ captions: caps });
        }
        break;
      case "voxa.turn_metrics": {
        const metrics = this.snap.metrics.slice();
        const i = metrics.findIndex((x) => x.gen === m.gen);
        if (i >= 0) metrics[i] = { ...metrics[i], ...m };
        else metrics.push(m);
        this.set({ metrics: metrics.slice(-30) });
        break;
      }
      case "voxa.handoff":
        this.set({ state: "handoff", handoff: m.reason });
        break;
    }
  }

  private send(topic: string, payload: unknown) {
    this.room?.localParticipant.publishData(enc.encode(JSON.stringify(payload)), { reliable: true, topic }).catch(() => {});
  }

  async setMuted(muted: boolean) {
    await this.room?.localParticipant.setMicrophoneEnabled(!muted);
    this.set({ muted });
  }

  sendCsat(score: number) { this.send("voxa.csat", { score }); }

  async end() {
    this.userEnded = true;
    await this.teardown();
    this.set({ state: "ended" });
  }

  private async teardown() {
    cancelAnimationFrame(this.raf);
    const r = this.room;
    this.room = null;
    this.audioEls.forEach((e) => e.remove());
    this.audioEls = [];
    await r?.disconnect().catch(() => {});
    await this.ctx?.close().catch(() => {});
    this.ctx = null;
    this.agentAnalyser = null;
  }

  private async onDisconnected(reason?: DisconnectReason): Promise<void> {
    if (this.userEnded || !this.room) return;
    if (reason === DisconnectReason.PARTICIPANT_REMOVED || reason === DisconnectReason.ROOM_DELETED) {
      // server ended the call (agent hung up / silence timeout): normal end, not an error
      await this.teardown();
      this.set({ state: "ended" });
      return;
    }
    const max = this.opts.maxRetries ?? 3;
    if (this.retries < max && this.creds) {
      this.retries++;
      this.set({ state: "reconnecting" });
      await new Promise((r) => setTimeout(r, 500 * 2 ** this.retries));
      try {
        await this.teardown();
        await this.connect(this.creds);
        return;
      } catch { /* fall through to next retry / failure */ }
      if (this.retries < max) return this.onDisconnected(reason);
    }
    this.fail(new Error("Connection lost. Please try again."));
  }

  private fail(e: unknown) {
    const err = e as Error & { name?: string };
    let msg = err?.message || "Something went wrong";
    if (err?.name === "NotAllowedError" || /permission/i.test(msg)) msg = "Microphone access was blocked. Allow the microphone in your browser's site settings and try again.";
    if (err?.name === "NotFoundError") msg = "No microphone was found on this device.";
    this.set({ state: "error", error: msg });
    void this.teardown();
  }
}

export { ConnectionState };

/** A human agent joining a waiting caller's room (warm transfer). Hears the caller; the AI agent leaves on join. */
export class HumanCall {
  private room = new Room();
  private els: HTMLMediaElement[] = [];
  summary = "";
  async join(creds: Credentials, onSummary: (s: string) => void, onEnd: () => void) {
    this.room.on(RoomEvent.TrackSubscribed, (t) => {
      if (t.kind !== Track.Kind.Audio) return;
      const el = t.attach() as HTMLMediaElement; document.body.appendChild(el); this.els.push(el);
    });
    this.room.on(RoomEvent.DataReceived, (d, _p, _k, topic) => {
      if (topic === "voxa.handoff_summary") { try { this.summary = JSON.parse(dec.decode(d)).summary; onSummary(this.summary); } catch { /* ignore */ } }
    });
    this.room.on(RoomEvent.Disconnected, () => { this.els.forEach((e) => e.remove()); onEnd(); });
    await this.room.connect(creds.url, creds.token);
    await this.room.localParticipant.setMicrophoneEnabled(true);
  }
  async setMuted(m: boolean) { await this.room.localParticipant.setMicrophoneEnabled(!m); }
  async leave() { await this.room.disconnect(); }
}
