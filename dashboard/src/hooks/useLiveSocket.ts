import { useEffect, useRef, useState } from 'react';
import { API_BASE_URL } from '../services/api';

export interface LiveLine {
  id: number;
  start: number;
  speaker: string;
  lang: string;
  source: string;
  translations: Record<string, string>;
  refined: boolean;
}

export interface DisplaySettings {
  font_size: number;
  lines: number;
  show_source: 'top' | 'bottom' | 'hidden';
  show_speaker: boolean;
  colour_speakers: boolean;
  theme: 'dark' | 'light';
}

const DEFAULT_DISPLAY: DisplaySettings = {
  font_size: 40,
  lines: 6,
  show_source: 'top',
  show_speaker: true,
  colour_speakers: true,
  theme: 'dark',
};

// API_BASE_URL is either '/api' (same origin) or 'http://host:port/api' (Vite dev server).
function socketUrl(): string {
  const base = API_BASE_URL.replace(/\/api$/, '');
  if (/^https?:/.test(base)) return `${base.replace(/^http/, 'ws')}/ws/live`;
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`;
}

/**
 * Live subtitle feed.
 *
 * The server sends `line` for a new utterance and `update` for one it has revised after seeing
 * what came next, so lines are keyed by id and replaced in place — appending an `update` would
 * show the same sentence twice.
 */
export function useLiveSocket() {
  const [lines, setLines] = useState<LiveLine[]>([]);
  const [display, setDisplay] = useState<DisplaySettings>(DEFAULT_DISPLAY);
  const [languages, setLanguages] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const retry = useRef<number | undefined>(undefined);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;

    const connect = () => {
      socket = new WebSocket(socketUrl());

      socket.onopen = () => setConnected(true);

      socket.onmessage = event => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'config') {
          setLanguages(msg.languages ?? []);
          if (msg.display) setDisplay({ ...DEFAULT_DISPLAY, ...msg.display });
          return;
        }
        if (msg.type === 'line' || msg.type === 'update') {
          setLines(prev => {
            const i = prev.findIndex(l => l.id === msg.line.id);
            if (i === -1) return [...prev, msg.line];
            const next = [...prev];
            next[i] = msg.line;
            return next;
          });
        }
      };

      socket.onclose = () => {
        setConnected(false);
        // The TV is unattended, so reconnect on its own rather than waiting for someone to reload.
        if (!closed) retry.current = window.setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry.current);
      socket?.close();
    };
  }, []);

  return { lines, display, languages, connected };
}
