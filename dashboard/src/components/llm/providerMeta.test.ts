import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PROVIDERS, metaOf, parseFallbackEntry } from './providerMeta.ts';

// server/llm.py DEFAULT_ENDPOINTS. A provider the backend can store but this page cannot name is
// what crashed the settings page: the stock config is 'anthropic', which this list used to omit.
const BACKEND_PROVIDERS = [
  'anthropic',
  'openai',
  'gemini',
  'groq',
  'ollama',
  'mistral',
  'openrouter',
  'nvidia_nim',
];

test('every provider the backend can default to has metadata here', () => {
  const known = new Set(PROVIDERS.map(p => p.value as string));
  const missing = BACKEND_PROVIDERS.filter(p => !known.has(p));
  assert.deepEqual(missing, [], `no metadata for: ${missing.join(', ')}`);
});

test('metaOf returns undefined instead of throwing on an unset or unknown provider', () => {
  assert.equal(metaOf(''), undefined);
  // Cast: the point is a value that reaches us at runtime despite the type.
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
