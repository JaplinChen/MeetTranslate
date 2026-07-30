import { request } from './http';

// =============================================================================
// Types
// =============================================================================

export interface ApiKey {
  id: string;
  name: string;
  keyPrefix: string;
  role: 'admin' | 'operator' | 'viewer';
  allowedIps?: string[];
  allowedSessions?: string[];
  isActive: boolean;
  expiresAt?: string;
  lastUsedAt?: string;
  usageCount: number;
  createdAt: string;
  apiKey?: string; // Only returned on creation
}

export interface AuditLog {
  id: string;
  action: string;
  severity: 'info' | 'warn' | 'error';
  apiKeyId?: string;
  apiKeyName?: string;
  sessionId?: string;
  sessionName?: string;
  ipAddress?: string;
  method?: string;
  path?: string;
  statusCode?: number;
  errorMessage?: string;
  createdAt: string;
}

export interface HealthStatus {
  status: 'ok' | 'error';
  timestamp?: string;
  /** Running backend version (from package.json) — read live so the sidebar never shows a stale build. */
  version?: string;
  details?: {
    database?: { status: string };
    redis?: { status: string };
    queue?: { status: string };
  };
}

export interface InfraStatus {
  // `builtIn` = OpenWA-Lab's own bundled container is actually running and backing this service (live),
  // not just the saved intent — falls back to the saved flag when Docker is unavailable. (#488)
  database: { connected: boolean; type: string; host: string; builtIn: boolean };
  redis: { enabled: boolean; connected: boolean; host: string; port: number; builtIn: boolean };
  queue: {
    enabled: boolean;
    webhooks: { pending: number; completed: number; failed: number };
  };
  storage: { type: 'local' | 's3'; path?: string; bucket?: string; builtIn: boolean; s3Available?: boolean };
  engine: {
    type: string;
    headless: boolean;
    // whatsapp-web.js only: the actual WhatsApp Web build in use (distinct from the library version)
    // and how it was chosen. (#488)
    webVersion?: string | null;
    webVersionSource?: 'pinned' | 'auto' | 'native';
  };
}

// Saved infrastructure config (from data/.env.generated) used to hydrate the form.
// Secrets are never returned — `*Set` flags indicate whether a value is stored.
export interface SavedConfig {
  database: {
    type: 'sqlite' | 'postgres';
    builtIn: boolean;
    host: string;
    port: string;
    username: string;
    database: string;
    schema: string;
    poolSize: number;
    sslEnabled: boolean;
    sslRejectUnauthorized: boolean;
    passwordSet: boolean;
  };
  redis: { enabled: boolean; builtIn: boolean; host: string; port: string; passwordSet: boolean };
  queue: { enabled: boolean };
  storage: {
    type: 'local' | 's3';
    builtIn: boolean;
    localPath: string;
    s3Bucket: string;
    s3Region: string;
    s3Endpoint: string;
    s3CredentialsSet: boolean;
  };
  engine: { type: string; headless: boolean; sessionDataPath: string; browserArgs: string };
}

export interface SaveConfigPayload {
  database?: {
    type: 'sqlite' | 'postgres';
    builtIn?: boolean;
    host?: string;
    port?: string;
    username?: string;
    password?: string;
    database?: string;
    schema?: string;
    poolSize?: number;
    sslEnabled?: boolean;
    sslRejectUnauthorized?: boolean;
  };
  redis?: {
    enabled?: boolean;
    builtIn?: boolean;
    host?: string;
    port?: string;
    password?: string;
  };
  queue?: {
    enabled?: boolean;
  };
  storage?: {
    type: 'local' | 's3';
    builtIn?: boolean;
    localPath?: string;
    s3Bucket?: string;
    s3Region?: string;
    s3AccessKey?: string;
    s3SecretKey?: string;
    s3Endpoint?: string;
  };
  engine?: {
    type?: string;
    headless?: boolean;
    sessionDataPath?: string;
    browserArgs?: string;
  };
}

export interface Settings {
  general: { apiBaseUrl: string; sessionTimeout: number; autoReconnect: boolean; debugMode: boolean };
  api: { rateLimit: number; rateLimitWindow: number; enableDocs: boolean };
  notifications: { emailEnabled: boolean; notificationEmail: string; webhookAlerts: boolean };
}

export type LlmProvider = 'ollama' | 'openai' | 'groq' | 'azure' | 'gemini';

export interface TranslateConfig {
  enabled: boolean;
  groupIds: string[];
  includeFromMe: boolean;
  minSendIntervalMs: number;
  notifyOnFailure: boolean;
  maxMessageLength: number;
  maxTranslationsPerMinute: number;
  llmProvider: LlmProvider;
  llmEndpoint: string;
  llmModel: string;
  llmApiKey: string;
  llmTemperature: number;
  llmFallbackModels: string[];
  llmPromptTemplate: string;
  llmPromptTemplateDefault?: string;
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

export interface GlossaryTerm {
  source: string; // 中文
  target: string; // 越南文
  count?: number; // 翻譯時實際套用次數
  category?: string; // 類別標籤（name/dept/term/asset/phrase/other）
}

export interface PendingGlossaryTerm {
  id: number;
  zh: string;
  vi: string;
  suggestedBy: string;
  at: string;
}

export interface TranslationCandidate {
  id: number;
  pairKey: string;
  source: string;
  translated: string;
  count: number;
  at: string;
}

export interface PhraseCandidate {
  id: number;
  phrase: string;
  translated: string;
  count: number;
  at: string;
}

export interface SenderEntry {
  jid: string; // @mention 的號碼 (digits)
  name: string; // 顯示名稱
  count?: number; // 翻譯時實際替換次數
}

// =============================================================================
// API Key API
// =============================================================================

export const apiKeyApi = {
  list: () => request<ApiKey[]>('/auth/api-keys'),
  get: (id: string) => request<ApiKey>(`/auth/api-keys/${id}`),
  create: (data: {
    name: string;
    role: string;
    allowedIps?: string[];
    allowedSessions?: string[];
    expiresAt?: string;
  }) =>
    request<ApiKey>('/auth/api-keys', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: Partial<ApiKey>) =>
    request<ApiKey>(`/auth/api-keys/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) => request<void>(`/auth/api-keys/${id}`, { method: 'DELETE' }),
  revoke: (id: string) => request<ApiKey>(`/auth/api-keys/${id}/revoke`, { method: 'POST' }),
};

// =============================================================================
// Audit/Logs API
// =============================================================================

export const auditApi = {
  list: (params?: { action?: string; severity?: string; limit?: number; offset?: number }) => {
    const query = new URLSearchParams();
    if (params?.action) query.set('action', params.action);
    if (params?.severity) query.set('severity', params.severity);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    const queryStr = query.toString();
    return request<{ data: AuditLog[]; total: number }>(`/audit${queryStr ? `?${queryStr}` : ''}`);
  },
};

// =============================================================================
// Health & Infrastructure API
// =============================================================================

export const healthApi = {
  check: () => request<HealthStatus>('/health'),
  ready: () => request<HealthStatus>('/health/ready'),
};

export const infraApi = {
  getStatus: () => request<InfraStatus>('/infra/status'),
  getConfig: () => request<SavedConfig>('/infra/config'),
  updateConfig: (config: Partial<InfraStatus>) =>
    request<InfraStatus>('/infra/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),
  saveConfig: (config: SaveConfigPayload) =>
    request<{ message: string; saved: boolean; envPath: string; profiles: string[] }>('/infra/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),
  restart: (profiles?: string[], profilesToRemove?: string[]) =>
    request<{
      message: string;
      restarting: boolean;
      profiles: string[];
      profilesToRemove: string[];
      estimatedTime: number;
    }>('/infra/restart', {
      method: 'POST',
      body: JSON.stringify({ profiles: profiles || [], profilesToRemove: profilesToRemove || [] }),
    }),
  healthCheck: () => request<{ status: string; timestamp: string }>('/infra/health'),
  // Data migration: export all Data-DB tables (call while still on the OLD database, before switching),
  // then import after the switch + restart. Used by the DB-switch migration guard so data isn't lost.
  exportData: () =>
    request<{ exportedAt: string; dataDbType: string; tables: Record<string, unknown[]>; counts: Record<string, number> }>(
      '/infra/export-data',
    ),
  importData: (tables: Record<string, unknown[]>) =>
    request<{ imported: boolean; counts?: Record<string, number>; message?: string; warnings?: string[] }>('/infra/import-data', {
      method: 'POST',
      body: JSON.stringify({ tables }),
    }),
};

// =============================================================================
// Translate API
// =============================================================================

export const translateApi = {
  getConfig: () => request<TranslateConfig>('/translate/config'),
  updateConfig: ({ llmPromptTemplateDefault: _readonly, apiKeySet: _mask, ...config }: Partial<TranslateConfig>) =>
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
  preview: (text: string, provider?: LlmProvider) =>
    request<{ pair: string; translated: string }>('/translate/preview', {
      method: 'POST',
      body: JSON.stringify({ text, ...(provider ? { provider } : {}) }),
    }),
  getGlossary: () => request<GlossaryTerm[]>('/translate/glossary'),
  addGlossaryTerm: (zh: string, vi: string, category?: string) =>
    request<GlossaryTerm[]>('/translate/glossary', {
      method: 'POST',
      body: JSON.stringify({ zh, vi, ...(category !== undefined ? { category } : {}) }),
    }),
  removeGlossaryTerm: (term: string) =>
    request<GlossaryTerm[]>(`/translate/glossary?term=${encodeURIComponent(term)}`, {
      method: 'DELETE',
    }),
  getMemoryCandidates: (limit = 20, offset = 0) =>
    request<{ items: TranslationCandidate[]; total: number }>(
      `/translate/memory/candidates?limit=${limit}&offset=${offset}`,
    ),
  approveMemoryCandidate: (id: number) =>
    request<TranslationCandidate[]>(`/translate/memory/${id}/approve`, { method: 'POST' }),
  dismissMemoryCandidate: (id: number) =>
    request<TranslationCandidate[]>(`/translate/memory/${id}`, { method: 'DELETE' }),
  getPhraseCandidates: () => request<PhraseCandidate[]>('/translate/memory/phrases'),
  scanPhraseCandidates: () => request<PhraseCandidate[]>('/translate/memory/phrases/scan', { method: 'POST' }),
  approvePhraseCandidate: (id: number) =>
    request<PhraseCandidate[]>(`/translate/memory/phrases/${id}/approve`, { method: 'POST' }),
  dismissPhraseCandidate: (id: number) =>
    request<PhraseCandidate[]>(`/translate/memory/phrases/${id}`, { method: 'DELETE' }),
  getPendingGlossary: () => request<PendingGlossaryTerm[]>('/translate/glossary/pending'),
  approvePendingGlossary: (id: number) =>
    request<void>(`/translate/glossary/pending/${id}/approve`, { method: 'POST' }),
  rejectPendingGlossary: (id: number) =>
    request<void>(`/translate/glossary/pending/${id}`, { method: 'DELETE' }),
  getCategories: () => request<string[]>('/translate/categories'),
  addCategory: (name: string) =>
    request<string[]>('/translate/categories', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),
  deleteCategory: (name: string) =>
    request<string[]>('/translate/categories', {
      method: 'DELETE',
      body: JSON.stringify({ name }),
    }),
  getSenders: () => request<SenderEntry[]>('/translate/senders'),
  addSender: (jid: string, name: string) =>
    request<SenderEntry[]>('/translate/senders', {
      method: 'POST',
      body: JSON.stringify({ jid, name }),
    }),
  removeSender: (jid: string) =>
    request<SenderEntry[]>(`/translate/senders?jid=${encodeURIComponent(jid)}`, {
      method: 'DELETE',
    }),
  importSenders: (sessionId: string) =>
    request<{ added: number; entries: SenderEntry[] }>('/translate/senders/import', {
      method: 'POST',
      body: JSON.stringify({ sessionId }),
    }),
  backfillSenders: (sessionId: string) =>
    request<{ filled: number; entries: SenderEntry[] }>('/translate/senders/backfill', {
      method: 'POST',
      body: JSON.stringify({ sessionId }),
    }),
};

// =============================================================================
// LLM key-proxy API (multi-key rotation for translate)
// =============================================================================

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

// =============================================================================
// Settings API
// =============================================================================

export const settingsApi = {
  get: () => request<Settings>('/settings'),
  update: (settings: Partial<Settings>) =>
    request<Settings>('/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
};
