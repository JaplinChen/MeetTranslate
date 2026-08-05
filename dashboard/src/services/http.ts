// API Service Layer for MeetTranslate Dashboard
// Single-user local app: the server listens on localhost only, so there is no auth header.

import { warnIfInsecureHttpUrl } from '../utils/urlSecurity.ts';

// Same-origin '/api' by default. Set VITE_API_URL to an API ORIGIN (e.g. http://localhost:8010)
// when the dashboard dev server and the Python service run on different ports.
//
// `env?.` because Vite defines `import.meta.env` and plain Node does not: this module runs under
// `node --test` too, where the bare property read threw at import time and took down any test that
// touched the service layer, whatever it was actually asserting.
const API_ORIGIN = (import.meta.env?.VITE_API_URL ?? '').replace(/\/+$/, '');
export const API_BASE_URL = `${API_ORIGIN}/api`;
if (API_ORIGIN) warnIfInsecureHttpUrl(API_ORIGIN, 'VITE_API_URL');

export async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const isFormData = options.body instanceof FormData;
  const headers: HeadersInit = {
    ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
    ...options.headers,
  };

  const response = await fetch(`${API_BASE_URL}${endpoint}`, { ...options, headers });

  if (!response.ok) {
    // On a non-JSON body fall through to `HTTP <status>` rather than statusText: the status code is
    // what the toast connection-lost de-dup matches on, and statusText is empty over HTTP/2 anyway.
    const error = await response.json().catch(() => ({}));
    const err = new Error(error.message || `HTTP ${response.status}`) as Error & { status?: number };
    err.status = response.status;
    throw err;
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}
