import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Single source of truth for the version shown in the dashboard: read it from
// package.json at build time so the sidebar always reflects the actual release
// (bumped via `npm version`), instead of a hard-coded literal that silently
// drifts. APP_VERSION env still overrides if explicitly provided.
const { version: pkgVersion } = JSON.parse(readFileSync(resolve(process.cwd(), 'package.json'), 'utf-8')) as {
  version: string;
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  appType: 'spa', // Enable SPA fallback for client-side routing
  define: {
    __APP_VERSION__: JSON.stringify(process.env.APP_VERSION || pkgVersion),
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
