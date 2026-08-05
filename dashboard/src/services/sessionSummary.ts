import type { SessionSummary } from './app.api';

/** What `/sessions` actually returns. A server older than the refine field omits it, which happens
 *  every time one is left running across an update. */
export type RawSessionSummary = Omit<SessionSummary, 'refine'> & Partial<Pick<SessionSummary, 'refine'>>;

/** Fills the field in so `SessionSummary` is true for every consumer. They all read `s.refine.state`
 *  on that promise — the session dropdown does it inside a map over every session, so one legacy row
 *  threw and the ErrorBoundary blanked the whole transcript. */
export const withRefine = (s: RawSessionSummary): SessionSummary => ({
  ...s,
  refine: s.refine ?? { state: 'idle', error: '' },
});
