import type { LlmProvider } from '../../services/api';

export interface ProviderMeta {
  label: string;
  endpoint: string;
  showEndpoint: boolean; // Ollama/OpenAI/Azure expose a server URL; Groq/Gemini have a fixed one.
  needsKey: boolean;
  apiKeyUrl?: string;
}

export const PROVIDERS: { value: LlmProvider; meta: ProviderMeta }[] = [
  { value: 'ollama', meta: { label: 'Ollama', endpoint: 'http://127.0.0.1:11434/api/chat', showEndpoint: true, needsKey: false } },
  { value: 'groq', meta: { label: 'Groq', endpoint: 'https://api.groq.com/openai/v1/chat/completions', showEndpoint: false, needsKey: true, apiKeyUrl: 'https://console.groq.com/keys' } },
  { value: 'openai', meta: { label: 'OpenAI', endpoint: 'https://api.openai.com/v1/chat/completions', showEndpoint: true, needsKey: true, apiKeyUrl: 'https://platform.openai.com/api-keys' } },
  { value: 'azure', meta: { label: 'Azure OpenAI', endpoint: 'https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-02-15-preview', showEndpoint: true, needsKey: true, apiKeyUrl: 'https://portal.azure.com' } },
  { value: 'gemini', meta: { label: 'Gemini', endpoint: 'https://generativelanguage.googleapis.com/v1beta', showEndpoint: false, needsKey: true, apiKeyUrl: 'https://aistudio.google.com/apikey' } },
];

export const metaOf = (p: LlmProvider): ProviderMeta => PROVIDERS.find(x => x.value === p)!.meta;

// The proxy routes "<provider>/<model>" (e.g. "gemini/flash"); the apply-key link should follow the
// model's real provider, not the OpenAI-compatible transport provider.
export function apiKeyUrlForModel(fallback: ProviderMeta, model: string): string | undefined {
  const slash = model.indexOf('/');
  const prefix = slash > 0 ? model.slice(0, slash) : '';
  return PROVIDERS.find(p => p.value === prefix)?.meta.apiKeyUrl ?? fallback.apiKeyUrl;
}

// One tab's provider config. provider '' means the (fallback) tab is unset — skipped on save.
// fallbackModels are extra models of THIS provider, tried after `model` before moving to the next tab.
export interface ProviderConfig {
  provider: LlmProvider | '';
  endpoint: string;
  model: string;
  apiKey: string;
  temperature: number;
  fallbackModels: string[];
}

export const emptyProviderConfig = (): ProviderConfig => ({ provider: '', endpoint: '', model: '', apiKey: '', temperature: 0, fallbackModels: [] });

const PROVIDER_VALUES = PROVIDERS.map(p => p.value);

/**
 * Parse a fallback entry into its resolved provider + model, mirroring the backend's resolveModel:
 * a "provider:model" prefix (provider ∈ known providers) crosses providers; anything else (incl. an
 * Ollama tag like "qwen3:8b") is a bare model on the active provider.
 */
export function parseFallbackEntry(
  entry: string,
  activeProvider: LlmProvider,
): { provider: LlmProvider; model: string } {
  const colon = entry.indexOf(':');
  const prefix = colon > 0 ? entry.slice(0, colon) : '';
  if (PROVIDER_VALUES.includes(prefix as LlmProvider)) {
    const model = entry.slice(colon + 1);
    if (model) return { provider: prefix as LlmProvider, model };
  }
  return { provider: activeProvider, model: entry };
}
