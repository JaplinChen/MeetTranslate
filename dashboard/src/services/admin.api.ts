/* Backend clients for the settings side of the app: health, the LLM/translate config, and the
   API-key rotation pool. The transcript, glossary and capture endpoints live in app.api.ts.

   This file arrived from another project and carried its API surface with it — key management,
   audit logs, infrastructure control, an app-settings store. None of it was ever called here and
   none of those endpoints exist in server/, so it is gone. What is left is what the pages
   actually reach for. */

import { request } from './http.ts';

/** What /api/health actually answers. It used to also declare a timestamp and a `details` block
 *  with database, redis and queue sub-states — none of which this app has, and none of which the
 *  endpoint has ever sent. */
export interface HealthStatus {
  status: 'ok' | 'error';
  /** Running backend version, read from the repo-root VERSION file the server also reads. */
  version?: string;
}

// Mirrors server/llm.py DEFAULT_ENDPOINTS, plus 'azure', which the backend accepts as a
// user-supplied endpoint without a default of its own. Keep the two lists in step: a provider the
// backend can store but this one omits reaches the settings page as an unknown value.
export type LlmProvider =
  | 'anthropic'
  | 'ollama'
  | 'openai'
  | 'groq'
  | 'azure'
  | 'gemini'
  | 'mistral'
  | 'openrouter'
  | 'nvidia_nim';

/** What /api/translate/config answers — exactly the keys llm.py's to_json() builds.
 *
 *  It used to declare nine more: enabled, groupIds, includeFromMe, minSendIntervalMs,
 *  notifyOnFailure, maxMessageLength, maxTranslationsPerMinute and the two llmPromptTemplate
 *  fields. All of them are message-relay settings from the project this dashboard came from, all
 *  were non-optional, and none has ever arrived. A non-optional field that is always undefined is
 *  the shape that cost a blank transcript page in #37. */
export interface TranslateConfig {
  llmProvider: LlmProvider;
  llmEndpoint: string;
  llmModel: string;
  /** Per-function model overrides; empty falls back to llmModel. */
  llmTranslateModel?: string;
  llmSummaryModel?: string;
  llmApiKey: string;
  llmTemperature: number;
  llmFallbackModels: string[];
  apiKeySet?: boolean;
  llmProviderConfigs: Record<string, LlmProviderSaved>;
}

export interface LlmProviderSaved {
  endpoint?: string;
  model?: string;
  apiKey?: string;
  apiKeySet?: boolean;
  temperature?: number;
  fallbackModels?: string[];
}

export interface LlmProbe {
  provider: LlmProvider;
  endpoint: string;
  model?: string;
  apiKey?: string;
}

export interface KeyStatus {
  provider: string;
  index: number;
  account: string;
  masked: string;
  status: string;
  requestCount: number;
  failureCount: number;
  /** Part of requestCount that bypassed the proxy (voice transcription calls the provider directly). */
  voiceRequestCount: number;
  /** Provider-reported ceiling, or null when the provider reports none (then show a bare count). */
  quota: {
    limitRequests: number;
    remainingRequests: number;
    limitAudioSeconds: number;
    remainingAudioSeconds: number;
  } | null;
}

export const healthApi = {
  check: () => request<HealthStatus>('/health'),
};

export const translateApi = {
  getConfig: () => request<TranslateConfig>('/translate/config'),
  // apiKeySet is a mask the server computes; sending it back would be meaningless.
  updateConfig: ({ apiKeySet: _mask, ...config }: Partial<TranslateConfig>) =>
    request<TranslateConfig>('/translate/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),
  testLlm: (probe: LlmProbe) =>
    request<{ ok: boolean; message: string }>('/translate/llm/test', {
      method: 'POST',
      body: JSON.stringify(probe),
    }),
  listLlmModels: (probe: LlmProbe) =>
    request<{ models: string[] }>('/translate/llm/models', {
      method: 'POST',
      body: JSON.stringify(probe),
    }),
};

export const keyProxyApi = {
  list: () => request<KeyStatus[]>('/keyproxy/keys'),
  add: (provider: string, apiKey: string, account: string) =>
    request<KeyStatus[]>('/keyproxy/keys', {
      method: 'POST',
      body: JSON.stringify({ provider, apiKey, account }),
    }),
  remove: (provider: string, index: number) =>
    request<KeyStatus[]>(`/keyproxy/keys/${encodeURIComponent(provider)}/${index}`, {
      method: 'DELETE',
    }),
};
