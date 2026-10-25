export type UiState =
  | "idle" | "connecting" | "listening" | "thinking" | "speaking" | "paused" | "handoff" | "reconnecting" | "ended" | "error";

export interface TurnMetrics {
  gen: number;
  voice_to_voice_ms: number | null;
  stt_ms: number | null;
  llm_ttft_ms: number | null;
  tts_ttfb_ms: number | null;
  /** Measured in the browser: speech-end event -> first audible agent audio (approximate, see docs). */
  client_heard_ms?: number | null;
}

export interface Caption {
  role: "user" | "agent";
  text: string;
  final: boolean;
  gen?: number;
}

export interface CallSnapshot {
  state: UiState;
  serverState: string;
  muted: boolean;
  captions: Caption[];
  metrics: TurnMetrics[];
  error?: string;
  handoff?: string;
  agentLevel: number;   // 0..1 output level for visualisation
  userLevel: number;
}

export interface Credentials { url: string; token: string; room?: string }
