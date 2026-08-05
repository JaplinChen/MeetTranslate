/**
 * Attributes that keep a browser or password manager out of a secret field.
 *
 * `autoComplete="off"` is not enough: Chrome ignores it on password fields and fills the site's
 * saved login credential in anyway, which then gets saved as the API key. `"new-password"` is the
 * opt-out it actually honours, and the `data-*` attrs are what 1Password and LastPass look for.
 *
 * Shared because it was learned once and then missed on the next field that needed it — spread it
 * into any input that takes a key rather than writing `autoComplete="off"` and assuming.
 */
export const NO_AUTOFILL = {
  autoComplete: 'new-password',
  'data-1p-ignore': true,
  'data-lpignore': 'true',
  'data-form-type': 'other',
} as const;
