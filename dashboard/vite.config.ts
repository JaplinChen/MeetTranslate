import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Single source of truth for the version shown in the dashboard, read at build time instead of a
// hard-coded literal that silently drifts. The repo-root VERSION file is authoritative because the
// server reads the same file for /api/health — and the sidebar prefers the server's answer, so two
// sources meant a bump in one of them changed nothing on screen. package.json is the fallback for
// a dashboard built on its own; APP_VERSION env still overrides both.
const readVersion = (): string => {
  try {
    return readFileSync(resolve(process.cwd(), '..', 'VERSION'), 'utf-8').trim();
  } catch {
    return (JSON.parse(readFileSync(resolve(process.cwd(), 'package.json'), 'utf-8')) as { version: string }).version;
  }
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  appType: 'spa', // Enable SPA fallback for client-side routing
  define: {
    __APP_VERSION__: JSON.stringify(process.env.APP_VERSION || readVersion()),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
  server: {
    port: 2886,
    proxy: {
      // The Python service; same port as the packaged app serves from.
      '/api': {
        target: 'http://localhost:8010',
        changeOrigin: true,
        secure: false,
      },
      // Subtitle stream. Without ws:true this silently falls back to a failed
      // HTTP upgrade and the live page shows nothing.
      '/ws': {
        target: 'http://localhost:8010',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
