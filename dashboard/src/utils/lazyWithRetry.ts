import { lazy, type ComponentType } from 'react';
import { loadChunkWithReload } from './chunkReload';

/**
 * Drop-in replacement for React.lazy that survives a stale-chunk failure after a redeploy: a failed
 * dynamic import() triggers a one-time full reload (fresh index.html + chunks) instead of bubbling to
 * the top-level ErrorBoundary and blanking the whole dashboard. See chunkReload.ts for the guard logic.
 */
// `name` scopes the one-shot reload flag to this chunk. Without it a page that loads two lazy
// chunks — every /settings/* route does — lets the one that succeeds clear the flag for the one
// that failed, and the reload repeats forever.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function lazyWithRetry<T extends ComponentType<any>>(name: string, factory: () => Promise<{ default: T }>) {
  return lazy(() =>
    loadChunkWithReload(
      factory,
      { reload: () => window.location.reload(), storage: window.sessionStorage },
      name,
    ),
  );
}
