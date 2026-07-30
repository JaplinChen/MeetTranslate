import { useState, useEffect, useCallback } from 'react';

export type Theme = 'light' | 'dark' | 'system' | 'anthropic' | 'anthropic-dark';
export type ThemePalette = 'openwa' | 'blue' | 'graphite' | 'indigo' | 'amber' | 'rose' | 'teal';

const THEME_KEY = 'openwalab_theme';
const PALETTE_KEY = 'openwalab_palette';

export const paletteOptions: Array<{ value: ThemePalette; label: string; color: string }> = [
  { value: 'openwa', label: 'OpenWA-Lab', color: '#25d366' },
  { value: 'blue', label: 'Blue', color: '#2563eb' },
  { value: 'graphite', label: 'Graphite', color: '#64748b' },
  { value: 'indigo', label: 'Indigo', color: '#4f46e5' },
  { value: 'amber', label: 'Amber', color: '#d97706' },
  { value: 'rose', label: 'Rose', color: '#e11d48' },
  { value: 'teal', label: 'Teal', color: '#0d9488' },
];

function isTheme(value: string | null): value is Theme {
  return (
    value === 'light' ||
    value === 'dark' ||
    value === 'system' ||
    value === 'anthropic' ||
    value === 'anthropic-dark'
  );
}

function isPalette(value: string | null): value is ThemePalette {
  return paletteOptions.some(option => option.value === value);
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    const saved = localStorage.getItem(THEME_KEY);
    return isTheme(saved) ? saved : 'anthropic';
  });
  const [palette, setPaletteState] = useState<ThemePalette>(() => {
    const saved = localStorage.getItem(PALETTE_KEY);
    return isPalette(saved) ? saved : 'openwa';
  });

  const applyTheme = useCallback((newTheme: Theme) => {
    const root = document.documentElement;

    if (newTheme === 'system') {
      // Remove data-theme to let CSS media query handle it
      root.removeAttribute('data-theme');
    } else {
      root.setAttribute('data-theme', newTheme);
    }
  }, []);

  const applyPalette = useCallback((newPalette: ThemePalette) => {
    document.documentElement.setAttribute('data-palette', newPalette);
  }, []);

  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme, applyTheme]);

  useEffect(() => {
    applyPalette(palette);
    localStorage.setItem(PALETTE_KEY, palette);
  }, [palette, applyPalette]);

  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme);
  }, []);

  const setPalette = useCallback((newPalette: ThemePalette) => {
    setPaletteState(newPalette);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState(prev => {
      if (prev === 'light') return 'dark';
      if (prev === 'dark') return 'system';
      return 'light';
    });
  }, []);

  // Get the resolved theme (what's actually displayed)
  const resolvedTheme =
    theme === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      : theme === 'anthropic'
        ? 'light'
        : theme === 'anthropic-dark'
          ? 'dark'
          : theme;

  return { theme, setTheme, toggleTheme, resolvedTheme, palette, setPalette, paletteOptions };
}
