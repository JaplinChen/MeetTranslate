import { request } from './http';

export interface DisplaySettings {
  font_size: number;
  lines: number;
  show_source: 'top' | 'bottom' | 'hidden';
  show_speaker: boolean;
  colour_speakers: boolean;
  theme: 'dark' | 'light';
}

/** A voice the room can name on sight, learned from someone naming a speaker once. */
export interface KnownSpeaker {
  name: string;
}

/** What the recogniser wrote against what was actually said, learned from an edit. */
export interface LearnedCorrection {
  wrong: string;
  right: string;
}

export interface AppConfig {
  languages: string[];
  inputDevice: string;
  whisperModel: string;
  availableModels: string[];
  pinnedLanguages: Record<string, string>;
  translatorReady: boolean;
  display: DisplaySettings;
}

export interface AudioDevice {
  index: number;
  name: string;
  channels: number;
  hostapi: string;
}

export interface GlossaryTerm {
  id: number;
  source: string;
  lang: string;
  /** translate = force a rendering, keep = never translate it, hint = bias ASR only. */
  mode: 'translate' | 'keep' | 'hint';
  category: string;
  targets: Record<string, string>;
}

export interface RecordingStatus {
  recording: boolean;
  path: string | null;
  seconds: number;
  /** Recent input level. Zero while recording means no audio is reaching the capture device. */
  peak: number;
  droppedBlocks: number;
  sessionId: number | null;
  backlog: number;
  errors: number;
}

export interface SessionSummary {
  id: number;
  started: string;
  ended: string | null;
  wav_path: string;
  lines: number;
}

export interface TranscriptLine {
  id: number;
  start: number;
  speaker: string;
  lang: string;
  source: string;
  refined: number;
  translations: Record<string, string>;
}

export const appApi = {
  getConfig: () => request<AppConfig>('/config'),
  putConfig: (body: Partial<AppConfig>) => request<AppConfig>('/config', { method: 'PUT', body: JSON.stringify(body) }),

  devices: () =>
    request<{ devices: AudioDevice[]; configured: string; selected: number | null; error: string | null }>('/devices'),

  glossary: () => request<GlossaryTerm[]>('/glossary'),
  addTerm: (body: Partial<GlossaryTerm>) =>
    request<GlossaryTerm[]>('/glossary', { method: 'POST', body: JSON.stringify(body) }),
  removeTerm: (source: string, lang = '') =>
    request<GlossaryTerm[]>(`/glossary?source=${encodeURIComponent(source)}&lang=${encodeURIComponent(lang)}`, {
      method: 'DELETE',
    }),

  startRecording: () => request<RecordingStatus>('/recording/start', { method: 'POST' }),
  stopRecording: () => request<RecordingStatus>('/recording/stop', { method: 'POST' }),
  recordingStatus: () => request<RecordingStatus>('/recording/status'),

  sessions: () => request<SessionSummary[]>('/sessions'),
  sessionLines: (id: number) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string> }>(`/sessions/${id}/lines`),
  setSpeakerNames: (id: number, names: Record<string, string>) =>
    request<Record<string, string>>(`/sessions/${id}/speakers`, { method: 'PUT', body: JSON.stringify(names) }),
  // Editing a line also teaches the correction: the backend stores what was written against what
  // was said, and applies it to every future transcript.
  // Asked before a term is added, not after: adding 料號 rewrote the real term 料耗 42 times
  // across seven interviews, and nothing said so.
  termCollisions: (source: string) =>
    request<{ source: string; collisions: { text: string; count: number }[] }>(
      `/glossary/collisions?source=${encodeURIComponent(source)}`),
  knownSpeakers: () => request<KnownSpeaker[]>('/speakers/known'),
  forgetSpeaker: (name: string) =>
    request<KnownSpeaker[]>(`/speakers/known/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  corrections: () => request<LearnedCorrection[]>('/corrections'),
  forgetCorrection: (wrong: string) =>
    request<LearnedCorrection[]>(`/corrections/${encodeURIComponent(wrong)}`, { method: 'DELETE' }),
  setLineSource: (id: number, lineId: number, source: string) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string> }>(
      `/sessions/${id}/lines/${lineId}`, { method: 'PUT', body: JSON.stringify({ source }) }),
};
