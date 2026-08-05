import type { SessionSummary } from './app.api.ts';

/** What `/sessions` actually returns. A server older than a given field omits it, which happens
 *  every time one is left running across an update. */
export type RawSessionSummary = Omit<SessionSummary, 'refine' | 'hasRecording'> &
  Partial<Pick<SessionSummary, 'refine' | 'hasRecording'>>;

/** Fills the field in so `SessionSummary` is true for every consumer. They all read `s.refine.state`
 *  on that promise — the session dropdown does it inside a map over every session, so one legacy row
 *  threw and the ErrorBoundary blanked the whole transcript. */
export const withRefine = (s: RawSessionSummary): SessionSummary => ({
  ...s,
  refine: s.refine ?? { state: 'idle', error: '' },
  // Assume present when the server does not say: an older backend has the audio it always had, and
  // guessing the other way would hide working playback behind a "recording missing" notice.
  hasRecording: s.hasRecording ?? true,
});
