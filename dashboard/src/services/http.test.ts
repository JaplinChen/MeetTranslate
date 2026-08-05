import { test } from 'node:test';
import assert from 'node:assert/strict';
import { API_BASE_URL, request } from './http.ts';
import { appApi } from './app.api.ts';

// Importing this module at all is half the point: `import.meta.env` is a Vite construct, and the
// bare `import.meta.env.VITE_API_URL` read ran at import time, so under `node --test` it threw
// before any assertion in any test that reached the service layer. These fail if that returns.

test('the service layer imports under plain node', () => {
  assert.equal(typeof request, 'function');
  assert.equal(typeof appApi.sessions, 'function');
});

test('with no VITE_API_URL set, the API is same-origin /api', () => {
  assert.equal(API_BASE_URL, '/api');
});
