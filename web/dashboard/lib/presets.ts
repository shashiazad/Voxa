/** Turn-taking presets shared by the agent editor and the onboarding wizard. */
export const PRESETS: Record<string, Record<string, number>> = {
  Relaxed: { endpoint_complete_ms: 500, endpoint_incomplete_ms: 2000, stage1_min_speech_ms: 220, vad_threshold_speaking: 0.7 },
  Balanced: { endpoint_complete_ms: 300, endpoint_incomplete_ms: 1500, stage1_min_speech_ms: 160, vad_threshold_speaking: 0.65 },
  Snappy: { endpoint_complete_ms: 220, endpoint_incomplete_ms: 1000, stage1_min_speech_ms: 130, vad_threshold_speaking: 0.6 },
};
