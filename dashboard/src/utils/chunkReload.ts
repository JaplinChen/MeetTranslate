// Recovery for a failed dynamic import() of a route/lazy chunk. The dominant cause is a redeploy:
// the running index.html references hashed chunk filenames that no longer exist on the server, so
// import() rejects. A one-time full reload pulls the fresh index + chunks. A sessionStorage flag
// guards against a reload loop when the failure is not deploy-related (adblock, offline, real 404).
// React-free on purpose so it is unit-testable without a DOM. See lazyWithRetry.ts for the wiring.
//
// The flag is per chunk. One shared key held up only while a page loaded a single chunk: /settings/*
// renders a lazy Settings parent with a lazy child in its Outlet, so the parent's success cleared the
// flag the failing child had just set, and the child earned a fresh reload every time round — the
// exact loop the flag exists to stop.
const RELOAD_KEY = 'mt_chunk_reloaded';

export interface ChunkReloadDeps {
  reload: () => void;
  storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
}

export async function loadChunkWithReload<T>(
  factory: () => Promise<T>,
  deps: ChunkReloadDeps,
  /** Identifies the chunk across a reload, so one chunk's recovery is not another's. */
  name = '',
): Promise<T> {
  const key = `${RELOAD_KEY}:${name}`;
  try {
    const mod = await factory();
    deps.storage.removeItem(key);
    return mod;
  } catch (err) {
    if (!deps.storage.getItem(key)) {
      deps.storage.setItem(key, '1');
      deps.reload();
      // Hold Suspense until the reload navigates away; never resolve/reject this load.
      return new Promise<T>(() => {});
    }
    // A reload already happened and it still failed → let the caller's error boundary surface it.
    throw err;
  }
}
