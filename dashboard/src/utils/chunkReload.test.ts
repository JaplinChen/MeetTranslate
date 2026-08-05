import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadChunkWithReload } from './chunkReload.ts';

function makeStorage(initial: Record<string, string> = {}): Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> {
  const m = new Map<string, string>(Object.entries(initial));
  return {
    getItem: k => m.get(k) ?? null,
    setItem: (k, v) => void m.set(k, v),
    removeItem: k => void m.delete(k),
  };
}

const flush = () => new Promise(resolve => setTimeout(resolve, 0));

test('returns the module and clears the reload flag on success', async () => {
  const storage = makeStorage({ 'mt_chunk_reloaded:Capture': '1' });
  const mod = { default: 'Component' };
  let reloads = 0;

  const result = await loadChunkWithReload(() => Promise.resolve(mod), { reload: () => reloads++, storage }, 'Capture');

  assert.equal(result, mod);
  assert.equal(reloads, 0);
  assert.equal(storage.getItem('mt_chunk_reloaded:Capture'), null);
});

test('reloads exactly once on a chunk failure when no reload has happened yet', async () => {
  const storage = makeStorage();
  let reloads = 0;

  // The result never settles (Suspense holds until the reload), so don't await it.
  void loadChunkWithReload(
    () => Promise.reject(new Error('Loading chunk 7 failed')),
    { reload: () => reloads++, storage },
    'Capture',
  );
  await flush();

  assert.equal(reloads, 1);
  assert.equal(storage.getItem('mt_chunk_reloaded:Capture'), '1');
});

test('rethrows instead of reloading again once a reload already happened (no loop)', async () => {
  const storage = makeStorage({ 'mt_chunk_reloaded:Capture': '1' });
  let reloads = 0;

  await assert.rejects(
    loadChunkWithReload(() => Promise.reject(new Error('still failing')), { reload: () => reloads++, storage }, 'Capture'),
    /still failing/,
  );
  assert.equal(reloads, 0);
});

// /settings/* renders a lazy Settings parent with a lazy child inside its Outlet, so two chunks
// load per boot, parent first. If a sibling success can clear the shared flag, the failing child
// gets a fresh reload every time round and the tab never stops reloading.
test('a sibling chunk succeeding does not buy the failing one another reload', async () => {
  const storage = makeStorage();
  let reloads = 0;
  const deps = { reload: () => reloads++, storage };

  // Returns what the child load did: 'pending' means it is holding Suspense for a reload,
  // 'threw' means it gave up and let the error boundary have it.
  const boot = async (): Promise<'pending' | 'threw'> => {
    await loadChunkWithReload(() => Promise.resolve({ default: 'Settings' }), deps, 'Settings');
    let outcome: 'pending' | 'threw' = 'pending';
    loadChunkWithReload(() => Promise.reject(new Error('chunk 9 failed')), deps, 'LlmSettings')
      .catch(() => { outcome = 'threw'; });
    await flush();
    return outcome;
  };

  assert.equal(await boot(), 'pending');   // first failure: reload, and hold Suspense meanwhile
  assert.equal(await boot(), 'threw');     // still failing after that reload: surface it, do not loop

  assert.equal(reloads, 1);
});
