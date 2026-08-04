import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { PROVIDERS, metaOf, parseFallbackEntry } from './providerMeta.ts';

// The settings page crashed on a stock install because the backend's default provider, 'anthropic',
// was not in PROVIDERS: metaOf found nothing and the caller dereferenced it. Read the backend's own
// table rather than copying it here, so this guard cannot drift the way the list it guards did.
const LLM_PY = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..', 'server', 'llm.py');

function backendProviders(): string[] {
  const src = readFileSync(LLM_PY, 'utf8');
  const block = src.match(/DEFAULT_ENDPOINTS\s*=\s*\{([\s\S]*?)\}/);
  assert.ok(block, `could not find DEFAULT_ENDPOINTS in ${LLM_PY}`);
  const names = [...block[1].matchAll(/^\s*"([^"]+)"\s*:/gm)].map(m => m[1]);
  assert.ok(names.length > 0, 'parsed DEFAULT_ENDPOINTS but found no providers — has the shape changed?');
  return names;
}

test('every provider the backend can default to has metadata here', () => {
  const known = new Set(PROVIDERS.map(p => p.value as string));
  const missing = backendProviders().filter(p => !known.has(p));
  assert.deepEqual(missing, [], `no metadata for: ${missing.join(', ')}`);
});

test('metaOf returns undefined instead of throwing on an unset or unknown provider', () => {
  assert.equal(metaOf(''), undefined);
  // The cast is the point: the backend stores llmProvider without validating it against a list,
  // so a value the type forbids really can arrive at runtime.
  assert.equal(metaOf('some-provider-we-have-never-heard-of' as never), undefined);
});

test('metaOf resolves a known provider', () => {
  assert.equal(metaOf('anthropic')?.label, 'Anthropic');
  assert.equal(metaOf('ollama')?.needsKey, false);
});

test('parseFallbackEntry splits a known provider prefix but keeps an ollama tag whole', () => {
  assert.deepEqual(parseFallbackEntry('gemini:flash', 'ollama'), { provider: 'gemini', model: 'flash' });
  assert.deepEqual(parseFallbackEntry('qwen3:8b', 'ollama'), { provider: 'ollama', model: 'qwen3:8b' });
});
